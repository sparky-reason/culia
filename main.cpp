#include "culia.h"
#include "julia_animator.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdlib.h>

#include <string>
#include <complex>

#include <GL/glew.h>

#include <SDL.h>
#include <SDL_opengl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;


int main()
{
	// initialise SDL
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
		MessageBox(NULL, (string("Error initializing SDL: ") + SDL_GetError()).c_str(), "Error", MB_OK | MB_ICONERROR);
		exit(-1);
	}
	atexit(SDL_Quit);

	// create OpenGL window
	SDL_Window* window = SDL_CreateWindow("Culia", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 0, 0,
		SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP);
	if (!window) {
		MessageBox(NULL, (string("Error creating SDL window: ") + SDL_GetError()).c_str(), "Error", MB_OK | MB_ICONERROR);
		exit(-1);
	}

	SDL_GLContext glcontext = SDL_GL_CreateContext(window);
	if (!glcontext) {
		MessageBox(NULL, (string("Error creating OpenGL context: ") + SDL_GetError()).c_str(), "Error", MB_OK | MB_ICONERROR);
		exit(-1);
	}

	// initialize GLEW
	GLenum glew_err = glewInit();
	if (GLEW_OK != glew_err) {
		MessageBox(NULL, "Error initializing GLEW", "Error", MB_OK | MB_ICONERROR);
		exit(-1);
	}

	// get window size
	int width, height;
	SDL_GetWindowSize(window, &width, &height);

	// create framebuffer with associated rederbuffer
	GLuint framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	GLuint renderbuffer;
	glGenRenderbuffers(1, &renderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);

	// register renderbuffer with CUDA
	cudaGraphicsResource_t cuda_renderbuffer;
	cudaError_t cuda_err = cudaGraphicsGLRegisterImage(&cuda_renderbuffer, renderbuffer, GL_RENDERBUFFER,
		cudaGraphicsRegisterFlagsWriteDiscard | cudaGraphicsRegisterFlagsSurfaceLoadStore);
	if (cuda_err != cudaSuccess) {
		MessageBox(NULL, (string("CUDA error: ") + cudaGetErrorString(cuda_err)).c_str(), "Error", MB_OK | MB_ICONERROR);
		exit(-1);
	}

	// hide mouse pointer
	SDL_ShowCursor(SDL_DISABLE);

	// main message loop
	JuliaAnimator julia_animator;
	SDL_Event event;
	for (;;) {
		unsigned int tick = SDL_GetTicks();

		// render Julia set to renderbuffer
		cuda_err = render_julia_set(cuda_renderbuffer, width, height, julia_animator.get_c(tick));
		if (cuda_err != cudaSuccess) {
			MessageBox(NULL, (string("CUDA error: ") + cudaGetErrorString(cuda_err)).c_str(), "Error", MB_OK | MB_ICONERROR);
			exit(-1);
		}

		// blit framebuffer
		glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST);

		// swap buffers
		SDL_GL_SwapWindow(window);

		// handle SDL events
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
			case SDL_KEYUP:
				if (event.key.keysym.sym == SDLK_ESCAPE) {
					SDL_Event quit;
					quit.type = SDL_QUIT;
					SDL_PushEvent(&quit);
				}
				break;

			case SDL_QUIT:
				SDL_GL_DeleteContext(glcontext);
				SDL_DestroyWindow(window);
				
				exit(0);
				break;
			}
		}
	}
}

#ifdef _WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	return main();
}
#endif
