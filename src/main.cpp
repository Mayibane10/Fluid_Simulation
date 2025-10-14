// glfw_glad_fluid_sim.cpp
// 2D Stable Fluid simulation with GLFW + GLAD
// Interactive: LMB inject density + velocity, RMB inject force, C clears

#include <bits/stdc++.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <omp.h>

// Simple grid-based stable fluid implementation (GPU)
//* Author: Mayiken Bellete - https://github.com/Mayibane10
//* Big thanks to: Sebastian Lague - https://github.com/seblague
//* Thanks for teaching me about Fluid sim
//* Watch his video abou Fluid simulation - https://www.youtube.com/watch?v=Q78wvrQ9xsU&t=263s

// Window
int WIN_W = 1300, WIN_H = 1300;

// Grid / sim params
int N = 275;           //! grid resolution (N x N)
float DT = 0.5f;       // Simulation timestep (delta time)
float DIFF = 1.0E-7f;  // Diffusion
float VISC = 4.0E-15f; // Viscosity
int LIN_ITERS = 6;     // iterations for linear solver

// Index helper
inline int IX(int x, int y, int Nlocal) { return x + y * (Nlocal + 2); }

struct Fluid
{
    int N;
    int size; // (N+2)^2
    float *s, *density;
    float *Vx, *Vy, *Vx0, *Vy0;
    float *temp;        // temporary buffer for solver
    unsigned char *img; // upload image buffer (RGB)
    Fluid(int N_) : N(N_)
    {
        size = (N + 2) * (N + 2);
        s = (float *)calloc(size, sizeof(float));
        density = (float *)calloc(size, sizeof(float));
        Vx = (float *)calloc(size, sizeof(float));
        Vy = (float *)calloc(size, sizeof(float));
        Vx0 = (float *)calloc(size, sizeof(float));
        Vy0 = (float *)calloc(size, sizeof(float));
        temp = (float *)calloc(size, sizeof(float));
        img = (unsigned char *)malloc((N + 2) * (N + 2) * 3);
    }
    ~Fluid()
    {
        free(s);
        free(density);
        free(Vx);
        free(Vy);
        free(Vx0);
        free(Vy0);
        free(temp);
        free(img);
    }
};

// Global fluid pointer and GL texture
Fluid *fluid = nullptr;
GLuint tex = 0;

// ---------- helper: enforce boundaries ----------
void set_bnd(int Nlocal, int b, float *x)
{
    // edges
    for (int i = 1; i <= Nlocal; i++)
    {
        x[IX(0, i, Nlocal)] = (b == 1) ? -x[IX(1, i, Nlocal)] : x[IX(1, i, Nlocal)];
        x[IX(Nlocal + 1, i, Nlocal)] = (b == 1) ? -x[IX(Nlocal, i, Nlocal)] : x[IX(Nlocal, i, Nlocal)];
        x[IX(i, 0, Nlocal)] = (b == 2) ? -x[IX(i, 1, Nlocal)] : x[IX(i, 1, Nlocal)];
        x[IX(i, Nlocal + 1, Nlocal)] = (b == 2) ? -x[IX(i, Nlocal, Nlocal)] : x[IX(i, Nlocal, Nlocal)];
    }
    // corners
    x[IX(0, 0, Nlocal)] = 0.5f * (x[IX(1, 0, Nlocal)] + x[IX(0, 1, Nlocal)]);
    x[IX(0, Nlocal + 1, Nlocal)] = 0.5f * (x[IX(1, Nlocal + 1, Nlocal)] + x[IX(0, Nlocal, Nlocal)]);
    x[IX(Nlocal + 1, 0, Nlocal)] = 0.5f * (x[IX(Nlocal, 0, Nlocal)] + x[IX(Nlocal + 1, 1, Nlocal)]);
    x[IX(Nlocal + 1, Nlocal + 1, Nlocal)] = 0.5f * (x[IX(Nlocal, Nlocal + 1, Nlocal)] + x[IX(Nlocal + 1, Nlocal, Nlocal)]);
}

// ---------- linear solver (parallel) ----------
// Solves: x = (x0 + a*(neighbors)) / c using iterations
void lin_solve_jacobi(int Nlocal, int b, float *x, float *x0, float a, float c, float *tempBuf)
{
    // We'll use tempBuf as 'x_old' and write into x as 'x_new'
    // initialize x_old = x
    memcpy(tempBuf, x, sizeof(float) * ((Nlocal + 2) * (Nlocal + 2)));

    for (int k = 0; k < LIN_ITERS; k++)
    {
#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 1; i <= Nlocal; i++)
        {
            for (int j = 1; j <= Nlocal; j++)
            {
                x[IX(i, j, Nlocal)] = (x0[IX(i, j, Nlocal)] + a * (tempBuf[IX(i - 1, j, Nlocal)] + tempBuf[IX(i + 1, j, Nlocal)] + tempBuf[IX(i, j - 1, Nlocal)] + tempBuf[IX(i, j + 1, Nlocal)])) / c;
            }
        }
        set_bnd(Nlocal, b, x);
        // swap buffers: new becomes old for next iter
        std::swap(x, tempBuf);
        // after swap, we want x to point to the "new" buffer for next write,
        // but since x was swapped, we continue — at end, if odd iterations,
        // x contains old data. To ensure correct final result in original x pointer,
        // if LIN_ITERS is odd, copy tempBuf -> x (we'll do unify at end)
    }
    // If number of iterations left us with result in tempBuf (because of final swap),
    // ensure original x points to latest result by copying if necessary:
    // Here we simply ensure x contains final by copying tempBuf into x if needed
    // Detect by comparing pointers? Simpler: run one final copy of tempBuf into x.
    // (cost is O(n), but small compared to solver)
    memcpy(x, tempBuf, sizeof(float) * ((Nlocal + 2) * (Nlocal + 2)));
    set_bnd(Nlocal, b, x);
}

// ---------- diffusion (calls linear solver) ----------
void diffuse(int Nlocal, int b, float *x, float *x0, float diff, float dt, float *temp)
{
    float a = dt * diff * Nlocal * Nlocal;
    lin_solve_jacobi(Nlocal, b, x, x0, a, 1 + 4 * a, temp);
}

// ---------- advection (parallel) ----------
void advect(int Nlocal, int b, float *d, float *d0, float *velocX, float *velocY, float dt)
{
    float dt0 = dt * Nlocal;
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Nlocal; i++)
    {
        for (int j = 1; j <= Nlocal; j++)
        {
            float x = i - dt0 * velocX[IX(i, j, Nlocal)];
            float y = j - dt0 * velocY[IX(i, j, Nlocal)];
            if (x < 0.5f)
                x = 0.5f;
            if (x > Nlocal + 0.5f)
                x = Nlocal + 0.5f;
            int i0 = (int)floorf(x);
            int i1 = i0 + 1;
            if (y < 0.5f)
                y = 0.5f;
            if (y > Nlocal + 0.5f)
                y = Nlocal + 0.5f;
            int j0 = (int)floorf(y);
            int j1 = j0 + 1;
            float s1 = x - i0;
            float s0 = 1 - s1;
            float t1 = y - j0;
            float t0 = 1 - t1;
            d[IX(i, j, Nlocal)] = s0 * (t0 * d0[IX(i0, j0, Nlocal)] + t1 * d0[IX(i0, j1, Nlocal)]) + s1 * (t0 * d0[IX(i1, j0, Nlocal)] + t1 * d0[IX(i1, j1, Nlocal)]);
        }
    }
    set_bnd(Nlocal, b, d);
}

// ---------- projection to make velocity field divergence-free ----------
void project(int Nlocal, float *velocX, float *velocY, float *p, float *div, float *temp)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Nlocal; i++)
    {
        for (int j = 1; j <= Nlocal; j++)
        {
            div[IX(i, j, Nlocal)] = -0.5f * (velocX[IX(i + 1, j, Nlocal)] - velocX[IX(i - 1, j, Nlocal)] + velocY[IX(i, j + 1, Nlocal)] - velocY[IX(i, j - 1, Nlocal)]) / Nlocal;
            p[IX(i, j, Nlocal)] = 0;
        }
    }
    set_bnd(Nlocal, 0, div);
    set_bnd(Nlocal, 0, p);
    lin_solve_jacobi(Nlocal, 0, p, div, 1.0f, 4.0f, temp);

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i <= Nlocal; i++)
    {
        for (int j = 1; j <= Nlocal; j++)
        {
            velocX[IX(i, j, Nlocal)] -= 0.5f * (p[IX(i + 1, j, Nlocal)] - p[IX(i - 1, j, Nlocal)]) * Nlocal;
            velocY[IX(i, j, Nlocal)] -= 0.5f * (p[IX(i, j + 1, Nlocal)] - p[IX(i, j - 1, Nlocal)]) * Nlocal;
        }
    }
    set_bnd(Nlocal, 1, velocX);
    set_bnd(Nlocal, 2, velocY);
}

// ---------- density step ----------
void density_step(int Nlocal, float *x, float *x0, float *Vx, float *Vy, float diff, float dt, float *temp)
{
    // swap x0 and x (caller already added sources)
    // We'll implement as copy swap: copy x into temp, then use diffuse etc.
    // But simpler: reuse pointers: we expect caller to have prepared x0 as previous state
    std::swap(x0, x);
    diffuse(Nlocal, 0, x, x0, diff, dt, temp);
    std::swap(x0, x);
    advect(Nlocal, 0, x, x0, Vx, Vy, dt);
}

// ---------- velocity step ----------
void velocity_step(int Nlocal, float *Vx, float *Vy, float *Vx0, float *Vy0, float visc, float dt, float *temp)
{
    std::swap(Vx0, Vx);
    diffuse(Nlocal, 1, Vx, Vx0, visc, dt, temp);
    std::swap(Vy0, Vy);
    diffuse(Nlocal, 2, Vy, Vy0, visc, dt, temp);
    project(Nlocal, Vx, Vy, Vx0, Vy0, temp);
    std::swap(Vx0, Vx);
    std::swap(Vy0, Vy);
    advect(Nlocal, 1, Vx, Vx0, Vx0, Vy0, dt);
    advect(Nlocal, 2, Vy, Vy0, Vx0, Vy0, dt);
    project(Nlocal, Vx, Vy, Vx0, Vy0, temp);
}

// ---------- OpenGL helpers ----------
void init_gl_texture(int Nlocal)
{
    if (tex == 0)
        glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // allocate initial texture once
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, Nlocal + 2, Nlocal + 2, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
}

// Upload density -> texture via glTexSubImage2D (faster)
void upload_density_to_texture(Fluid *f)
{
    int W = f->N + 2;
    int H = f->N + 2;
    unsigned char *img = f->img;
    float *density = f->density;
    int Nlocal = f->N;

// fill in parallel
#pragma omp parallel for schedule(static)
    for (int j = 0; j < H; j++)
    {
        for (int i = 0; i < W; i++)
        {
            float d = density[IX(i, j, Nlocal)];
            if (d < 0.0f)
                d = 0.0f;
            if (d > 1.0f)
                d = 1.0f;
            unsigned char v = (unsigned char)(255.0f * d);
            int idx = 3 * (i + j * W);
            img[idx + 0] = v;
            img[idx + 1] = v;
            img[idx + 2] = v;
        }
    }

    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGB, GL_UNSIGNED_BYTE, img);
}

// Render textured full-screen quad (immediate mode for simplicity)
void render_texture()
{
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(-1, -1);
    glTexCoord2f(1, 0);
    glVertex2f(1, -1);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

// ---------- input handlers ----------
double lastX = 0, lastY = 0;

void add_density_at(Fluid *f, int x, int y, float amount)
{
    if (x < 1 || x > f->N || y < 1 || y > f->N)
        return;
    f->density[IX(x, y, f->N)] += amount;
}

void add_velocity_at(Fluid *f, int x, int y, float amountX, float amountY)
{
    if (x < 1 || x > f->N || y < 1 || y > f->N)
        return;
    f->Vx[IX(x, y, f->N)] += amountX;
    f->Vy[IX(x, y, f->N)] += amountY;
}

void cursor_pos_callback(GLFWwindow *window, double xpos, double ypos)
{
    if (!fluid)
        return;
    int winW, winH;
    glfwGetWindowSize(window, &winW, &winH);
    int i = (int)((xpos / winW) * fluid->N + 1);
    int j = (int)(((winH - ypos) / winH) * fluid->N + 1);
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        add_density_at(fluid, i, j, 50.0f * DT);
        float dx = (xpos - lastX), dy = (ypos - lastY);
        add_velocity_at(fluid, i, j, dx * 0.1f, -dy * 0.1f);
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
    {
        float fx = (xpos - lastX) * 0.5f;
        float fy = (lastY - ypos) * 0.5f;
        add_velocity_at(fluid, i, j, fx, fy);
    }
    lastX = xpos;
    lastY = ypos;
}

void scroll_callback(GLFWwindow *window, double xoff, double yoff)
{
    if (yoff > 0)
    {
        if (N < 512)
            N *= 2;
    }
    else
    {
        if (N > 16)
            N /= 2;
    }
    // re-create fluid and texture
    if (fluid)
    {
        delete fluid;
        fluid = new Fluid(N);
        init_gl_texture(N);
    }
}

// ---------- main ----------
int main()
{
    if (!glfwInit())
        return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    GLFWwindow *window = glfwCreateWindow(WIN_W, WIN_H, "2D Fluid renderer (fast)", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        fprintf(stderr, "Failed to init GLAD\n");
        return -1;
    }

    // init sim
    fluid = new Fluid(N);
    init_gl_texture(N);

    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);

    double prevTime = glfwGetTime();
    double lastSimTime = prevTime;
    const double simHz = 60.0;
    const double simDt = 1.0 / simHz;

    // set number of threads (optional)
    // omp_set_num_threads(omp_get_max_threads()); // default is fine

    while (!glfwWindowShouldClose(window))
    {
        double cur = glfwGetTime();
        double elapsed = cur - prevTime;
        prevTime = cur;

        // cap sim steps to ~60 Hz
        if (cur - lastSimTime >= simDt)
        {
            lastSimTime = cur;

// ambient decay (parallel)
#pragma omp parallel for schedule(static)
            for (int i = 0; i < fluid->size; i++)
            {
                fluid->density[i] *= 0.995f;
            }

            // sim steps (we pass fluid->temp for temporary buffer)
            velocity_step(fluid->N, fluid->Vx, fluid->Vy, fluid->Vx0, fluid->Vy0, VISC, DT, fluid->temp);
            density_step(fluid->N, fluid->density, fluid->s, fluid->Vx, fluid->Vy, DIFF, DT, fluid->temp);
        }

        // upload and render
        upload_density_to_texture(fluid);

        glViewport(0, 0, WIN_W, WIN_H);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        render_texture();

        glfwSwapBuffers(window);
        glfwPollEvents();

        // keyboard handling
        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS)
        {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < fluid->size; i++)
            {
                fluid->density[i] = 0.0f;
                fluid->Vx[i] = 0.0f;
                fluid->Vy[i] = 0.0f;
            }
        }
    }

    delete fluid;
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
