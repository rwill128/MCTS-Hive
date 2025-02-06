import numpy as np
import moderngl
import moderngl_window as mglw


class AlgaeSimulation(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Algae Simulation"
    fullscreen = True         # Opens the window in fullscreen mode.
    resizable = True          # Allow window resizes if needed.
    window_size = (800, 600)  # Fallback size if fullscreen is not available.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Use the actual window size for the simulation.
        self.simulation_size = self.wnd.size  # (width, height)
        print("Simulation size:", self.simulation_size)

        # Initialize simulation state: black everywhere (0.0) with a few seeded algae pixels (value 1.0)
        initial_state = np.zeros(self.simulation_size, dtype=np.float32)
        num_seeds = 100  # for example, 100 random seeds
        xs = np.random.randint(0, self.simulation_size[0], size=num_seeds)
        ys = np.random.randint(0, self.simulation_size[1], size=num_seeds)
        initial_state[xs, ys] = 1.0  # seed algae

        # Create textures with 32-bit float data.
        self.tex1 = self.ctx.texture(
            self.simulation_size, 1, initial_state.tobytes(),
            dtype='f4'
        )
        self.tex2 = self.ctx.texture(
            self.simulation_size, 1, None,
            dtype='f4'
        )
        # Use NEAREST filtering for a crisp cell-by-cell update.
        self.tex1.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex2.filter = (moderngl.NEAREST, moderngl.NEAREST)
        # Clamp to edge so that sampling off the edge doesn’t wrap.
        self.tex1.repeat_x = self.tex1.repeat_y = False
        self.tex2.repeat_x = self.tex2.repeat_y = False

        # Start with tex1 as the current state and tex2 as the target.
        self.current_tex = self.tex1
        self.next_tex = self.tex2

        # Create a framebuffer that renders into next_tex.
        self.fbo = self.ctx.framebuffer(color_attachments=[self.next_tex])

        # Shader program that updates the simulation state.
        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_text;
                out vec2 v_text;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D state;
                in vec2 v_text;
                out float fragColor;

                const float growth_threshold = 0.2;

                void main(){
                    float current = texture(state, v_text).r;

                    float sum = 0.0;
                    int count = 0;
                    ivec2 texSize = textureSize(state, 0);
                    vec2 pixelSize = 1.0 / vec2(texSize);
                    for (int i = -1; i <= 1; i++){
                        for (int j = -1; j <= 1; j++){
                            if(i == 0 && j == 0) continue;
                            vec2 offset = vec2(float(i), float(j)) * pixelSize;
                            sum += texture(state, v_text + offset).r;
                            count++;
                        }
                    }
                    float neighbor_avg = sum / float(count);

                    float sun = mix(0.2, 1.0, v_text.y);

                    if(current > 0.0){
                        fragColor = 1.0;
                    } else if(neighbor_avg > growth_threshold && sun > 0.5){
                        fragColor = 1.0;
                    } else {
                        fragColor = 0.0;
                    }
                }
            '''
        )

        # Set up a fullscreen quad (two triangles) to cover the viewport.
        vertices = np.array([
            # positions    # texture coordinates
            -1.0, -1.0,    0.0, 0.0,
            1.0, -1.0,    1.0, 0.0,
            -1.0,  1.0,    0.0, 1.0,
            -1.0,  1.0,    0.0, 1.0,
            1.0, -1.0,    1.0, 0.0,
            1.0,  1.0,    1.0, 1.0,
        ], dtype='f4')
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert', 'in_text')

        # Shader for displaying the simulation state with a yellowish background.
        self.display_prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                in vec2 in_text;
                out vec2 v_text;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_text = in_text;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform sampler2D state;
                in vec2 v_text;
                out vec4 fragColor;
                void main(){
                    float s = texture(state, v_text).r;
                    
                    // Compute a background color that is subtly yellow
                    // based on the vertical coordinate (to simulate sunlight).
                    vec4 bgColor = vec4(0.1 + 0.3 * v_text.y,
                                        0.1 + 0.3 * v_text.y,
                                        0.0,
                                        1.0);
                    
                    // Mix the background with bright green for algae.
                    fragColor = mix(bgColor, vec4(0.0, 1.0, 0.0, 1.0), s);
                }
            '''
        )
        self.display_vao = self.ctx.simple_vertex_array(
            self.display_prog, self.vbo, 'in_vert', 'in_text'
        )

    def on_render(self, time, frame_time):
        # === Simulation Step ===
        self.current_tex.use(location=0)
        self.prog['state'] = 0  # our simulation shader uses texture unit 0

        self.fbo.use()
        self.vao.render(moderngl.TRIANGLES)

        # Swap the textures (ping-pong).
        self.current_tex, self.next_tex = self.next_tex, self.current_tex
        self.fbo = self.ctx.framebuffer(color_attachments=[self.next_tex])

        # === Display Step ===
        self.ctx.screen.use()
        self.current_tex.use(location=0)
        self.display_prog['state'] = 0
        self.display_vao.render(moderngl.TRIANGLES)


if __name__ == '__main__':
    mglw.run_window_config(AlgaeSimulation)
