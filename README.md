# 📐 Differential Equation Solver
A powerful, interactive Python application for solving ordinary differential equations (ODEs), partial differential equations (PDEs), and their systems with built-in visualization capabilities.

(c) 2026, Michael Stal


LICENSE used: MIT


## Versions
- solver.py:        Regular solver with support for PDEs and ODEs
- solver_plus.py:   Additional experimental support of Systems of ODEs and Systems  of PDEs 

## 🌟 Features
Core Capabilities

- ✅ Single ODEs - First, second, third order differential equations
- ✅ Systems of ODEs - Coupled differential equations (predator-prey, oscillators, etc.)
- ✅ Single PDEs - Heat equation, wave equation, and more
- ✅ Systems of PDEs - Coupled partial differential equations
- ✅ Initial Conditions - Apply boundary and initial conditions
- ✅ Visualization - Plot solutions, derivatives, and phase portraits
- ✅ Symbolic Solutions - Exact analytical solutions using SymPy

## User Experience

- 🎯 Interactive command-line interface
- 📚 Built-in examples and help system
- 🔍 Detailed error messages and guidance
- 📊 Beautiful matplotlib visualizations
- 🚀 Easy-to-use notation system


## 📦 Installation
### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies
- pip install sympy numpy matplotlib

### Download the Application
```
# Clone or download the repository
git clone https://github.com/ms1963/de-solver.git
cd de-solver

# Or download solver.py and solver_plus.py directly
```

## 🚀 Quick Start
Run the Application
python solver.py  OR python solver_plus.py

### Basic Usage
Command: ode
ODE: y'' + y = 0
Initial conditions: y(0)=1, y'(0)=0

✓ Solution: y(x) = cos(x)

Command: visualize


## 📖 Notation Guide
For ODEs (Ordinary Differential Equations)


```
Notation
Meaning
Example



y'
First derivative dy/dx
y' = 2*y


y''
Second derivative d²y/dx²
y'' + y = 0


y'''
Third derivative d³y/dx³
y''' + y' = x
```

For PDEs (Partial Differential Equations)


```
Notation
Meaning
Example



u_t
∂u/∂t
u_t = alpha*u_xx


u_x
∂u/∂x
u_x + u_t = 0


u_xx
∂²u/∂x²
u_tt = c**2*u_xx


u_tt
∂²u/∂t²
u_tt - u_xx = 0
```

Mathematical Functions


```
Function
Notation
Example



Exponential
exp(x)
y' = exp(x)


Sine
sin(x)
y'' + sin(x) = 0


Cosine
cos(x)
y = cos(x)


Natural Log
log(x)
y' = log(x)


Square Root
sqrt(x)
y = sqrt(x)


Pi
pi
y = sin(pi*x)


Euler's Number
e or E
y = E**x

```

## 📝 Examples
### 1️⃣ Simple Harmonic Oscillator

```
Problem: Undamped oscillation
Command: ode
ODE: y'' + y = 0
Initial conditions: y(0)=1, y'(0)=0

Solution:
y(x) = cos(x)

Physical Meaning: A mass on a spring with no damping

```
### 2️⃣ Exponential Growth

```
Problem: Population growth model
Command: ode
ODE: y' = 2*y
Initial conditions: y(0)=1

Solution:
y(x) = exp(2*x)

Physical Meaning: Unrestricted population growth with rate constant 2
```

### 3️⃣ Damped Oscillator

```
Problem: Forced damped oscillation
Command: ode
ODE: y'' + 2*y' + 2*y = exp(-x)*sin(x)
Initial conditions: y(0)=1, y'(0)=1

Solution:
y(x) = e^(-x)*[(1 - x/2)*cos(x) + 2.5*sin(x)]

Physical Meaning: Damped spring-mass system with external forcing
```

### 4️⃣ Circular Motion (System of ODEs)
```
Problem: Uniform circular motion
Command: ode_sys
Number of equations: 2
Equation 1: x' = -y
Equation 2: y' = x
Initial conditions: x(0)=1, y(0)=0

Solution:
x(t) = cos(t)
y(t) = sin(t)

Physical Meaning: Particle moving in a circle with unit radius
Visualization: Creates a beautiful phase portrait showing circular motion!

```

### 5️⃣ Predator-Prey Model (Lotka-Volterra)
```
Problem: Population dynamics
Command: ode_sys
Number of equations: 2
Equation 1: x' = x - x*y
Equation 2: y' = -y + x*y
Initial conditions: x(0)=2, y(0)=1

Physical Meaning: 

x = prey population
y = predator population
Shows cyclical population dynamics
```

### 6️⃣ Heat Equation (PDE)
```
Problem: Heat diffusion in 1D
Command: pde
PDE: u_t = alpha*u_xx

Physical Meaning: Temperature distribution over time in a rod
```

### 7️⃣ Wave Equation (PDE)
```
Problem: Wave propagation
Command: pde
PDE: u_tt = c**2*u_xx

Physical Meaning: Vibrating string or sound wave propagation
```

### 8️⃣ First Order Linear ODE
```
Problem: Basic linear differential equation
Command: ode
ODE: y' + y = x
Initial conditions: y(0)=1

Solution:
y(x) = x - 1 + 2*exp(-x)
```

### 9️⃣ Second Order with Constant Coefficients
```
Problem: Homogeneous ODE
Command: ode
ODE: y'' - 3*y' + 2*y = 0
Initial conditions: y(0)=1, y'(0)=0

Solution:
y(x) = 2*exp(x) - exp(2*x)
```

### 🔟 Coupled Harmonic Oscillators
```
Problem: Two coupled springs
Command: ode_sys
Number of equations: 2
Equation 1: x'' + 2*x - y = 0
Equation 2: y'' + 2*y - x = 0
Initial conditions: x(0)=1, y(0)=0, x'(0)=0, y'(0)=0

Physical Meaning: Two masses connected by springs
```

## 🎨 Visualization Features
### Single ODE Visualization
When you solve an ODE and type visualize, you get 3 plots:

```
Solution y(x) - The main solution curve
First Derivative y'(x) - Rate of change
Second Derivative y''(x) - Acceleration/curvature

Example:
Command: ode
ODE: y'' + 2*y' + 2*y = exp(-x)*sin(x)
Initial conditions: y(0)=1, y'(0)=1

Command: visualize

Result: Beautiful damped oscillation with exponential envelope!
```

### System of ODEs Visualization
For 2D systems, you get:
```
x(t) vs t - First function over time
y(t) vs t - Second function over time
Phase Portrait - Trajectory in x-y plane

Example:
Command: ode_sys
Number of equations: 2
Equation 1: x' = -y
Equation 2: y' = x
Initial conditions: x(0)=1, y(0)=0

Command: visualize

Result: Perfect circle in phase space! 🎯
```

### 🛠️ Commands Reference
Main Commands



Command: Description



- ode: Solve a single ordinary differential equation
- ode_sys: Solve a system of ODEs
- pde: Solve a single partial differential equation
- pde_sys: Solve a system of PDEs
- visualize: Visualize the last solution
- examples: Show detailed example problems
- help: Show notation guide and help
- exit: Exit the program



## 💡 Tips &amp; Best Practices
### Writing Equations
#### ✅ DO:
- Use explicit multiplication: 2*x not 2x
- Use parentheses for clarity: (x+1)/(x-1)
- Separate initial conditions with commas: y(0)=1, y'(0)=0

#### ❌ DON'T:
- Forget multiplication signs: 2x ❌ (use 2*x ✅)
- Use implicit parentheses: 1/2*x might be ambiguous
- Mix up derivative notation: dy/dx ❌ (use y' ✅)

### Initial Conditions
Number of ICs = Number of Constants

- First-order ODE: 1 IC (e.g., y(0)=1)
- Second-order ODE: 2 ICs (e.g., y(0)=1, y'(0)=0)
- System of 2 ODEs: 2 ICs (e.g., x(0)=1, y(0)=0)


Format:
- y(0)=1          # Value at x=0
- y'(0)=0         # Derivative at x=0
- y''(0)=2        # Second derivative at x=0



## Visualization

✅ Solutions must have no undetermined constants to visualize
✅ Provide initial conditions for specific solutions
✅ General solutions (with C1, C2, etc.) cannot be plotted


## 🔬 Advanced Examples
Nonlinear ODE
```
Command: ode
ODE: y' = y**2 - x
Initial conditions: y(0)=1
```
Higher Order ODE
```
Command: ode
ODE: y''' + y'' + y' + y = 0
Initial conditions: y(0)=1, y'(0)=0, y''(0)=0
```
Coupled System with 3 Equations
```
Command: ode_sys
Number of equations: 3
Equation 1: x' = -y
Equation 2: y' = x - z
Equation 3: z' = y
Initial conditions: x(0)=1, y(0)=0, z(0)=0
```

Reaction-Diffusion PDE System
```
Command: pde_sys
Number of equations: 2
Equation 1: u_t = D1*u_xx + u - u*v
Equation 2: v_t = D2*v_xx + u*v - v
```

## 📊 Output Examples
Successful Solution
```
======================================================================
RESULT
======================================================================

✓ Solution: y(x) = exp(2*x)

======================================================================

System Solution
======================================================================
SYSTEM SOLUTION
======================================================================

✓ x(t) = cos(t)
✓ y(t) = sin(t)

======================================================================

With Undetermined Constants
======================================================================
RESULT
======================================================================

✓ Solution: y(x) = C1*exp(x) + C2*exp(-x)

⚠ Warning: Solution contains undetermined constants: {C1, C2}
  Provide initial conditions for a specific solution.

======================================================================
```

## 🐛 Troubleshooting
Common Issues

### Issue: "Could not parse equation"
Cause: Syntax error in equation
Solution:

Check for explicit multiplication: 2*x not 2x
Verify parentheses are balanced
Use correct function names: exp(x) not e^x

Example:
❌ y' = 2x + sin x
✅ y' = 2*x + sin(x)


### Issue: "Solution contains undetermined constants"
Cause: Not enough initial conditions
Solution:

Provide as many ICs as the order of the ODE
Second-order ODE needs 2 ICs: y(0)=1, y'(0)=0

Example:
❌ y'' + y = 0 with y(0)=1 only
✅ y'' + y = 0 with y(0)=1, y'(0)=0


### Issue: "Cannot visualize"
Cause: Solution has undetermined constants
Solution:

Add initial conditions to get specific solution
General solutions (with C1, C2) cannot be plotted


### Issue: "Unknown function in system"
Cause: Function name mismatch
Solution:

Ensure function names are consistent
Use simple names: x, y, z (not x1, x2)

Example:
❌ Equation 1: x' = -y, Equation 2: z' = x    # z not defined!

✅ Equation 1: x' = -y, Equation 2: y' = x


## 🎓 Educational Use Cases
For Students

- ✅ Verify homework solutions
- ✅ Visualize solution behavior
- ✅ Understand phase portraits
- ✅ Learn ODE/PDE concepts interactively

For Teachers

- ✅ Generate examples for lectures
- ✅ Create visualizations for presentations
- ✅ Demonstrate solution techniques
- ✅ Show real-world applications

For Researchers

- ✅ Quick analytical solutions
- ✅ Model verification
- ✅ Parameter exploration
- ✅ Visualization of dynamics


## 🔧 Technical Details
Solver Backend

- Symbolic Engine: SymPy 1.12+
- Numerical Computation: NumPy
- Visualization: Matplotlib
- Methods: dsolve() for ODEs, pdsolve() for PDEs
- Separation of variables
- Laplace transforms (when applicable)



Supported Equation Types
ODEs

- ✅ First-order linear and nonlinear
- ✅ Second-order linear with constant coefficients
- ✅ Higher-order ODEs
- ✅ Systems of linear ODEs
- ✅ Some nonlinear systems

PDEs

- ✅ Heat equation (parabolic)
- ✅ Wave equation (hyperbolic)
- ✅ Laplace equation (elliptic)
- ✅ Some separable PDEs

Limitations

- ⚠️ Not all PDEs have analytical solutions
- ⚠️ Nonlinear systems may not be solvable symbolically
- ⚠️ Complex boundary conditions may require numerical methods
- ⚠️ Very stiff equations may need specialized solvers


## 📚 Further Reading
- SymPy Documentation

- SymPy ODE Solver
- SymPy PDE Solver

Differential Equations Resources

- MIT OpenCourseWare - Differential Equations
- Paul's Online Math Notes
- Khan Academy - Differential Equations

## Applications

- Physics: Newton's laws, wave propagation, heat transfer
- Biology: Population dynamics, epidemiology (SIR models)
- Engineering: Control systems, circuit analysis
- Economics: Growth models, market dynamics


## 🤝 Contributing
Contributions are welcome! Here's how you can help:
Reporting Bugs

Check if the issue already exists
Provide a minimal example that reproduces the bug
Include error messages and system information

Suggesting Features

New equation types to support
Improved visualization options
Additional examples
Documentation improvements

Code Contributions

Fork the repository
Create a feature branch
Make your changes
Test thoroughly
Submit a pull request


## 📄 License
```
This project is licensed under the MIT License - see the LICENSE file for details.
MIT License

Copyright (c) 2026, Miochael Stal,   Differential Equation Solver

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- SymPy Team - For the powerful symbolic mathematics library
- NumPy Team - For numerical computation capabilities
- Matplotlib Team - For beautiful visualizations
- Contributors - Everyone who has helped improve this project


## 📞 Contact &amp; Support
Get Help

📖 Read the documentation (this README)
💬 Type help in the application
📝 Type examples for detailed examples
🐛 Report bugs via GitHub Issues

Stay Updated

- ⭐ Star the repository
- 👀 Watch for updates
- 🔔 Enable notifications


## 🎯 Quick Reference Card
Most Common Commands
# Solve simple ODE
Command: ode
ODE: y' = 2*y
Initial conditions: y(0)=1

# Solve system
Command: ode_sys
Number of equations: 2
Equation 1: x' = -y
Equation 2: y' = x
ICs: x(0)=1, y(0)=0

# Visualize
Command: visualize

# Get help
Command: help

# See examples
Command: examples

# Exit
Command: exit

## Notation Cheat Sheet



You Want
Type This

```
dy/dx
y'


d²y/dx²
y''


e^x
exp(x)


∂u/∂t
u_t


∂²u/∂x²
u_xx


π
pi


e
e or E
```


## 🌟 Star History
If you find this project useful, please consider giving it a star! ⭐

Happy Solving! 🎉

Made with ❤️ for students, educators, and researchers

Version: 2.0, Last Updated: January 2026, Python Version: 3.7+
