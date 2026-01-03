"""
Differential Equation Solver -  Experimental Version
Supports ODEs, PDEs, and their systems with comprehensive visualization
Version: 4.0 (Complete but Experimental)

Features:
- Symbolic and numerical ODE/PDE solving
- Systems of ODEs and PDEs
- Boundary and initial condition support
- 2D/3D visualization with export
- Comprehensive error handling
- Security hardened
- Works in all environments
- Export to JSON, LaTeX, CSV
- Animation support
- Parameter sweeps
- Stability analysis

Author: Michael Stal
License: MIT
Date: 2026-01-03
"""

import sympy as sp
from sympy import symbols, Function, Eq, dsolve, sympify
from sympy import exp, sin, cos, tan, log, sqrt, pi, E, I, oo
from sympy import Derivative, Integral, simplify, expand, factor
from sympy.parsing.sympy_parser import (
    parse_expr, 
    standard_transformations, 
    implicit_multiplication_application
)

import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.optimize import fsolve

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

import re
import warnings
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import traceback
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class SolutionExporter:
    """Handle solution export to various formats"""
    
    @staticmethod
    def export_to_json(solution_dict: Dict, filename: str) -> bool:
        """Export solution to JSON file"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'type': solution_dict.get('type', 'unknown'),
                'solution': str(solution_dict.get('solution', '')),
                'equation': str(solution_dict.get('equation_str', '')),
                'metadata': {
                    'solver_version': '4.0',
                    'method': solution_dict.get('method', 'symbolic')
                }
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"❌ Error exporting to JSON: {str(e)}")
            return False
    
    @staticmethod
    def export_to_latex(solution_dict: Dict, filename: str) -> bool:
        """Export solution to LaTeX file"""
        try:
            from sympy import latex
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("\\documentclass{article}\n")
                f.write("\\usepackage{amsmath}\n")
                f.write("\\usepackage{amssymb}\n")
                f.write("\\begin{document}\n\n")
                f.write("\\section*{Differential Equation Solution}\n\n")
                f.write(f"\\textbf{{Date:}} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if 'equation_str' in solution_dict:
                    f.write(f"\\subsection*{{Equation}}\n")
                    f.write(f"\\texttt{{{solution_dict['equation_str']}}}\n\n")
                
                if 'solution' in solution_dict:
                    sol = solution_dict['solution']
                    f.write("\\subsection*{Solution}\n")
                    
                    if isinstance(sol, list):
                        for i, s in enumerate(sol):
                            f.write("\\begin{equation}\n")
                            f.write(latex(s))
                            f.write("\n\\end{equation}\n\n")
                    elif isinstance(sol, (Eq, sp.Basic)):
                        f.write("\\begin{equation}\n")
                        f.write(latex(sol))
                        f.write("\n\\end{equation}\n\n")
                    else:
                        f.write(f"\\texttt{{{str(sol)}}}\n\n")
                
                f.write("\\end{document}\n")
            
            return True
        except Exception as e:
            print(f"❌ Error exporting to LaTeX: {str(e)}")
            return False
    
    @staticmethod
    def export_to_csv(data: Dict, filename: str) -> bool:
        """Export numerical data to CSV"""
        try:
            if 'numerical_data' not in data:
                print("⚠ No numerical data to export")
                return False
            
            num_data = data['numerical_data']
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header
                if 'header' in num_data:
                    writer.writerow(num_data['header'])
                
                # Write data
                if 'rows' in num_data:
                    writer.writerows(num_data['rows'])
            
            return True
        except Exception as e:
            print(f"❌ Error exporting to CSV: {str(e)}")
            return False


class InputValidator:
    """Validate and sanitize user inputs"""
    
    @staticmethod
    def validate_equation(eq_str: str) -> Tuple[bool, str]:
        """Validate equation string"""
        if not eq_str or not eq_str.strip():
            return False, "Empty equation"
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'os\.',
            r'sys\.',
            r'subprocess',
            r'__.*__'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, eq_str, re.IGNORECASE):
                return False, f"Potentially dangerous pattern: {pattern}"
        
        # Check for balanced parentheses
        if eq_str.count('(') != eq_str.count(')'):
            return False, "Unbalanced parentheses"
        
        # Check for balanced brackets
        if eq_str.count('[') != eq_str.count(']'):
            return False, "Unbalanced brackets"
        
        return True, "Valid"
    
    @staticmethod
    def validate_number(num_str: str, allow_negative: bool = True, 
                       allow_zero: bool = True) -> Tuple[bool, Optional[float]]:
        """Validate and parse number string"""
        try:
            num = float(num_str)
            
            if not allow_negative and num < 0:
                return False, None
            if not allow_zero and num == 0:
                return False, None
            if not np.isfinite(num):
                return False, None
            
            return True, num
        except (ValueError, TypeError):
            return False, None
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        filename = os.path.basename(filename)
        filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        return filename


class NumericalSolver:
    """Numerical methods for ODEs and PDEs"""
    
    @staticmethod
    def solve_ode_numerical(func, y0: Union[float, List[float]], 
                           t_span: Tuple[float, float], 
                           t_eval: Optional[np.ndarray] = None,
                           method: str = 'RK45') -> Dict:
        """
        Solve ODE numerically using scipy
        
        Args:
            func: Function dy/dt = func(t, y)
            y0: Initial conditions (scalar or array)
            t_span: (t_start, t_end)
            t_eval: Time points for evaluation
            method: Integration method (RK45, RK23, DOP853, etc.)
        
        Returns:
            Dictionary with solution or error message
        """
        try:
            # Ensure y0 is a list
            if not isinstance(y0, (list, np.ndarray)):
                y0 = [y0]
            
            if t_eval is None:
                t_eval = np.linspace(t_span[0], t_span[1], 1000)
            
            sol = solve_ivp(
                func, 
                t_span, 
                y0, 
                t_eval=t_eval, 
                method=method,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10
            )
            
            if not sol.success:
                return {'success': False, 'message': sol.message}
            
            return {
                'success': True,
                't': sol.t,
                'y': sol.y,
                'sol': sol,
                'method': method
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    @staticmethod
    def solve_heat_equation_numerical(
        alpha: float, 
        L: float, 
        T: float,
        nx: int = 100, 
        nt: int = 200,
        initial_condition=None,
        boundary_conditions=None
    ) -> Dict:
        """
        Solve 1D heat equation: u_t = alpha * u_xx
        
        Args:
            alpha: Thermal diffusivity
            L: Domain length
            T: Time duration
            nx: Number of spatial points
            nt: Number of time points
            initial_condition: Function u(x, 0)
            boundary_conditions: Dict with 'left' and 'right' values
        
        Returns:
            Dictionary with solution or error
        """
        try:
            x = np.linspace(0, L, nx)
            t = np.linspace(0, T, nt)
            dx = L / (nx - 1)
            dt = T / (nt - 1)
            
            # Stability condition
            r = alpha * dt / (dx**2)
            if r > 0.5:
                return {
                    'success': False,
                    'message': f'Unstable: r={r:.3f} > 0.5. Reduce dt or increase dx.'
                }
            
            u = np.zeros((nt, nx))
            
            # Initial condition
            if initial_condition is None:
                u[0, :] = np.sin(np.pi * x / L)
            else:
                u[0, :] = initial_condition(x)
            
            # Boundary conditions
            if boundary_conditions is None:
                boundary_conditions = {'left': 0, 'right': 0}
            
            # Time stepping
            for n in range(0, nt - 1):
                for i in range(1, nx - 1):
                    u[n + 1, i] = u[n, i] + r * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])
                
                u[n + 1, 0] = boundary_conditions.get('left', 0)
                u[n + 1, -1] = boundary_conditions.get('right', 0)
            
            return {
                'success': True,
                'x': x,
                't': t,
                'u': u,
                'r': r,
                'method': 'explicit_fd'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    @staticmethod
    def solve_wave_equation_numerical(
        c: float, 
        L: float, 
        T: float,
        nx: int = 100, 
        nt: int = 200,
        initial_position=None,
        initial_velocity=None
    ) -> Dict:
        """
        Solve 1D wave equation: u_tt = c^2 * u_xx
        
        Args:
            c: Wave speed
            L: Domain length
            T: Time duration
            nx: Number of spatial points
            nt: Number of time points
            initial_position: Function u(x, 0)
            initial_velocity: Function u_t(x, 0)
        
        Returns:
            Dictionary with solution or error
        """
        try:
            x = np.linspace(0, L, nx)
            t = np.linspace(0, T, nt)
            dx = L / (nx - 1)
            dt = T / (nt - 1)
            
            # CFL condition
            r = c * dt / dx
            if r > 1:
                return {
                    'success': False,
                    'message': f'Unstable: CFL={r:.3f} > 1. Reduce dt or increase dx.'
                }
            
            u = np.zeros((nt, nx))
            
            # Initial position
            if initial_position is None:
                u[0, :] = np.sin(np.pi * x / L)
            else:
                u[0, :] = initial_position(x)
            
            # Initial velocity
            if initial_velocity is None:
                u[1, :] = u[0, :]
            else:
                v0 = initial_velocity(x)
                u[1, 1:-1] = u[0, 1:-1] + dt * v0[1:-1] + \
                             0.5 * r**2 * (u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2])
            
            # Time stepping
            for n in range(1, nt - 1):
                u[n + 1, 1:-1] = 2 * u[n, 1:-1] - u[n - 1, 1:-1] + \
                                 r**2 * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2])
                
                u[n + 1, 0] = 0
                u[n + 1, -1] = 0
            
            return {
                'success': True,
                'x': x,
                't': t,
                'u': u,
                'r': r,
                'method': 'explicit_fd'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}
    
    @staticmethod
    def solve_reaction_diffusion_numerical(
        D1: float,
        D2: float,
        alpha: float,
        beta: float,
        L: float,
        T: float,
        nx: int = 100,
        nt: int = 200
    ) -> Dict:
        """
        Solve reaction-diffusion system:
        u_t = D1*u_xx + alpha*u - beta*u*v
        v_t = D2*v_xx + beta*u*v - alpha*v
        
        Returns:
            Dictionary with solution or error
        """
        try:
            x = np.linspace(0, L, nx)
            t = np.linspace(0, T, nt)
            dx = L / (nx - 1)
            dt = T / (nt - 1)
            
            # Stability check
            r1 = D1 * dt / (dx**2)
            r2 = D2 * dt / (dx**2)
            
            if r1 > 0.5 or r2 > 0.5:
                return {
                    'success': False,
                    'message': f'Unstable: r1={r1:.3f}, r2={r2:.3f}. Reduce dt.'
                }
            
            u = np.zeros((nt, nx))
            v = np.zeros((nt, nx))
            
            # Initial conditions (random perturbation)
            u[0, :] = 1.0 + 0.1 * np.random.randn(nx)
            v[0, :] = 1.0 + 0.1 * np.random.randn(nx)
            
            # Time stepping
            for n in range(0, nt - 1):
                for i in range(1, nx - 1):
                    # Diffusion + reaction for u
                    u_diff = r1 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
                    u_react = dt * (alpha * u[n, i] - beta * u[n, i] * v[n, i])
                    u[n+1, i] = u[n, i] + u_diff + u_react
                    
                    # Diffusion + reaction for v
                    v_diff = r2 * (v[n, i+1] - 2*v[n, i] + v[n, i-1])
                    v_react = dt * (beta * u[n, i] * v[n, i] - alpha * v[n, i])
                    v[n+1, i] = v[n, i] + v_diff + v_react
                
                # Periodic boundary conditions
                u[n+1, 0] = u[n+1, -2]
                u[n+1, -1] = u[n+1, 1]
                v[n+1, 0] = v[n+1, -2]
                v[n+1, -1] = v[n+1, 1]
            
            return {
                'success': True,
                'x': x,
                't': t,
                'u': u,
                'v': v,
                'method': 'explicit_fd'
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}


class StabilityAnalyzer:
    """Analyze stability of differential equations"""
    
    @staticmethod
    def analyze_ode_equilibria(rhs_func, var_names: List[str], 
                               search_range: Tuple[float, float] = (-10, 10)) -> Dict:
        """
        Find and analyze equilibrium points for ODE system
        
        Args:
            rhs_func: Function that returns derivatives
            var_names: List of variable names
            search_range: Range to search for equilibria
        
        Returns:
            Dictionary with equilibrium points and stability
        """
        try:
            n_vars = len(var_names)
            
            # Find equilibria
            def equations(y):
                return rhs_func(0, y)
            
            equilibria = []
            
            # Try multiple initial guesses
            for _ in range(20):
                x0 = np.random.uniform(search_range[0], search_range[1], n_vars)
                try:
                    sol = fsolve(equations, x0, full_output=True)
                    if sol[2] == 1:  # Solution found
                        eq_point = sol[0]
                        
                        # Check if already found
                        is_new = True
                        for existing in equilibria:
                            if np.allclose(eq_point, existing, atol=1e-6):
                                is_new = False
                                break
                        
                        if is_new:
                            equilibria.append(eq_point)
                except:
                    pass
            
            # Analyze stability
            results = []
            for eq in equilibria:
                # Compute Jacobian numerically
                epsilon = 1e-7
                jacobian = np.zeros((n_vars, n_vars))
                
                for i in range(n_vars):
                    perturb = np.zeros(n_vars)
                    perturb[i] = epsilon
                    
                    f_plus = rhs_func(0, eq + perturb)
                    f_minus = rhs_func(0, eq - perturb)
                    
                    jacobian[:, i] = (f_plus - f_minus) / (2 * epsilon)
                
                # Eigenvalues
                eigenvalues = np.linalg.eigvals(jacobian)
                
                # Stability
                max_real = np.max(np.real(eigenvalues))
                if max_real < 0:
                    stability = "Stable"
                elif max_real > 0:
                    stability = "Unstable"
                else:
                    stability = "Marginally Stable"
                
                results.append({
                    'point': eq,
                    'eigenvalues': eigenvalues,
                    'stability': stability
                })
            
            return {
                'success': True,
                'equilibria': results,
                'count': len(results)
            }
        except Exception as e:
            return {'success': False, 'message': str(e)}


# ============================================================================
# MAIN SOLVER CLASS
# ============================================================================

class DifferentialEquationSolver:
    """Main solver class for differential equations"""
    
    def __init__(self):
        self.last_solution = None
        self.last_type = None
        self.last_equations = None
        self.output_dir = "de_solutions"
        self.validator = InputValidator()
        self.exporter = SolutionExporter()
        self.numerical_solver = NumericalSolver()
        self.stability_analyzer = StabilityAnalyzer()
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create directory for saving outputs"""
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            print(f"✓ Output directory: {os.path.abspath(self.output_dir)}/")
        except Exception as e:
            print(f"⚠ Warning: Could not create directory: {str(e)}")
            self.output_dir = "."
    
    def run(self):
        """Main application loop"""
        self.print_banner()
        
        while True:
            self.print_menu()
            
            try:
                command = input("\n🔹 Command: ").strip().lower()
                
                if command == 'ode':
                    self.solve_ode()
                elif command == 'ode_sys':
                    self.solve_ode_system()
                elif command == 'pde':
                    self.solve_pde()
                elif command == 'pde_sys':
                    self.solve_pde_system()
                elif command == 'numerical':
                    self.solve_numerical()
                elif command == 'visualize':
                    self.visualize()
                elif command == 'animate':
                    self.create_animation()
                elif command == 'stability':
                    self.analyze_stability()
                elif command == 'export':
                    self.export_solution()
                elif command == 'examples':
                    self.show_examples()
                elif command == 'help':
                    self.show_help()
                elif command == 'exit':
                    print("\n👋 Thank you for using the Differential Equation Solver!")
                    print(f"📁 Your files: {os.path.abspath(self.output_dir)}/")
                    break
                else:
                    print(f"\n❌ Unknown command: '{command}'")
                    print("   Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n\n⚠ Interrupted by user")
                confirm = input("Exit? (y/n): ").strip().lower()
                if confirm == 'y':
                    break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                if input("Show traceback? (y/n): ").strip().lower() == 'y':
                    traceback.print_exc()
    
    def print_banner(self):
        """Print application banner"""
        print("=" * 80)
        print(" DIFFERENTIAL EQUATION SOLVER v4.0 - Experimental".center(80))
        print("=" * 80)
        print("\n✨ Complete Features:")
        print("  • Symbolic &amp; Numerical ODE/PDE solving")
        print("  • Systems of equations (ODEs &amp; PDEs)")
        print("  • 2D/3D visualization &amp; animation")
        print("  • Stability analysis")
        print("  • Export (JSON, LaTeX, CSV)")
        print("  • Production-grade error handling")
        print("  • Works in all environments")
        print("=" * 80)
        print(f"\n📁 Output: {os.path.abspath(self.output_dir)}/")
        print("=" * 80)
    
    def print_menu(self):
        """Print command menu"""
        print("\n" + "=" * 80)
        print("COMMANDS")
        print("=" * 80)
        print("  ode        - Solve single ODE (symbolic)")
        print("  ode_sys    - Solve system of ODEs")
        print("  pde        - Solve single PDE")
        print("  pde_sys    - Solve system of PDEs")
        print("  numerical  - Numerical solver")
        print("  visualize  - Visualize last solution")
        print("  animate    - Create animation")
        print("  stability  - Stability analysis")
        print("  export     - Export solution")
        print("  examples   - Show examples")
        print("  help       - Show help")
        print("  exit       - Exit program")
        print("=" * 80)
    
    def generate_filename(self, prefix: str = "solution", 
                         extension: str = "png") -> str:
        """Generate unique filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        counter = 0
        
        while True:
            if counter == 0:
                filename = f"{prefix}_{timestamp}.{extension}"
            else:
                filename = f"{prefix}_{timestamp}_{counter}.{extension}"
            
            filepath = os.path.join(self.output_dir, filename)
            
            if not os.path.exists(filepath):
                return filepath
            
            counter += 1
            if counter > 100:
                raise RuntimeError("Could not generate unique filename")
    
    def save_plot(self, fig, prefix: str = "solution"):
        """Save plot to file"""
        try:
            filename = self.generate_filename(prefix, "png")
            
            fig.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"\n✅ Plot saved:")
                print(f"   📄 {os.path.abspath(filename)}")
                print(f"   📊 {file_size / 1024:.2f} KB")
            else:
                print(f"\n⚠ Warning: File not found after save")
                
        except Exception as e:
            print(f"\n❌ Error saving plot: {str(e)}")
        finally:
            plt.close(fig)
    
    # ========================================================================
    # ODE SOLVER
    # ========================================================================
    
    def solve_ode(self):
        """Solve a single ODE"""
        print("\n" + "=" * 80)
        print("SOLVE ODE (Symbolic)")
        print("=" * 80)
        
        print("\n📖 Notation:")
        print("  y', y'', y'''  → Derivatives")
        print("  exp(x), sin(x), cos(x), log(x), sqrt(x)")
        print("  pi, e")
        
        print("\n💡 Examples:")
        print("  y' = 2*y")
        print("  y'' + y = 0")
        print("  y'' + 2*y' + y = exp(x)")
        
        ode_str = input("\n🔹 Enter ODE: ").strip()
        
        if not ode_str:
            print("❌ No equation entered")
            return
        
        is_valid, msg = self.validator.validate_equation(ode_str)
        if not is_valid:
            print(f"❌ Invalid equation: {msg}")
            return
        
        print("\n📋 Initial conditions (optional):")
        print("  Examples: y(0)=1  or  y(0)=1, y'(0)=0")
        ics_str = input("  ICs: ").strip()
        
        try:
            solution = self._solve_ode_internal(ode_str, ics_str)
            
            if solution is None:
                print("\n❌ Could not solve equation")
                return
            
            self.last_solution = solution
            self.last_type = 'ode'
            self.last_equations = ode_str
            
            self._display_ode_solution(solution)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if input("Show details? (y/n): ").strip().lower() == 'y':
                traceback.print_exc()
    
    def _solve_ode_internal(self, ode_str: str, ics_str: str) -> Optional[Dict]:
        """Internal ODE solver"""
        try:
            x = symbols('x', real=True)
            y = Function('y')
            
            ode_eq = self._parse_ode(ode_str, x, y)
            if ode_eq is None:
                return None
            
            print("\n" + "-" * 80)
            print("🔄 Solving...")
            print("-" * 80)
            
            try:
                general_solution = dsolve(ode_eq, y(x))
                print(f"✓ General solution found")
                
                if isinstance(general_solution, list):
                    general_solution = general_solution[0]
                
            except Exception as e:
                print(f"⚠ Symbolic solution failed: {str(e)}")
                print("💡 Try 'numerical' solver")
                return None
            
            if ics_str:
                solution = self._apply_ode_ics(general_solution, ics_str, x, y)
            else:
                solution = general_solution
            
            return {
                'solution': solution,
                'variable': x,
                'function': y,
                'equation': ode_eq,
                'equation_str': ode_str,
                'type': 'ode'
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def _parse_ode(self, ode_str: str, x, y) -> Optional[Eq]:
        """Parse ODE string"""
        try:
            ode_str = ode_str.replace("y'''", "Derivative(y(x), x, x, x)")
            ode_str = ode_str.replace("y''", "Derivative(y(x), x, x)")
            ode_str = ode_str.replace("y'", "Derivative(y(x), x)")
            ode_str = re.sub(r'\by\b(?!\()', 'y(x)', ode_str)
            
            if '=' in ode_str:
                parts = ode_str.split('=')
                if len(parts) != 2:
                    print("❌ Equation must have exactly one '='")
                    return None
                left, right = parts
            else:
                left = ode_str
                right = '0'
            
            namespace = {
                'x': x, 'y': y, 'exp': exp, 'sin': sin, 'cos': cos,
                'tan': tan, 'log': log, 'sqrt': sqrt, 'pi': pi,
                'e': E, 'E': E, 'I': I, 'Derivative': Derivative, 'oo': oo
            }
            
            try:
                transformations = standard_transformations + (implicit_multiplication_application,)
                left_expr = parse_expr(left.strip(), local_dict=namespace, 
                                      transformations=transformations)
                right_expr = parse_expr(right.strip(), local_dict=namespace, 
                                       transformations=transformations)
            except Exception as e:
                print(f"❌ Parse error: {str(e)}")
                return None
            
            return Eq(left_expr, right_expr)
            
        except Exception as e:
            print(f"❌ Could not parse: {str(e)}")
            return None
    
    def _apply_ode_ics(self, general_solution, ics_str: str, x, y) -> Eq:
        """Apply initial conditions"""
        print("\n" + "=" * 80)
        print("APPLYING INITIAL CONDITIONS")
        print("=" * 80)
        
        try:
            if isinstance(general_solution, Eq):
                rhs = general_solution.rhs
            else:
                rhs = general_solution
            
            constants = sorted(list(rhs.free_symbols - {x}), key=str)
            
            if not constants:
                print("✓ No constants to determine")
                return general_solution
            
            print(f"📊 Constants: {', '.join(map(str, constants))}")
            
            ics = []
            ic_parts = [ic.strip() for ic in ics_str.split(',') if ic.strip()]
            
            for ic_str_part in ic_parts:
                match = re.match(r"y(\'+)?\(([^)]+)\)\s*=\s*(.+)", ic_str_part.strip())
                if match:
                    deriv_order = len(match.group(1)) if match.group(1) else 0
                    point_str = match.group(2)
                    value_str = match.group(3)
                    
                    try:
                        point = float(sympify(point_str))
                        value = float(sympify(value_str))
                        ics.append((deriv_order, point, value))
                        
                        deriv_notation = "y" + "'" * deriv_order if deriv_order > 0 else "y"
                        print(f"  ✓ {deriv_notation}({point}) = {value}")
                    except Exception as e:
                        print(f"  ⚠ Could not parse '{ic_str_part}': {str(e)}")
            
            if len(ics) == 0:
                return general_solution
            
            if len(ics) < len(constants):
                print(f"\n⚠ Warning: {len(constants)} constants, {len(ics)} ICs")
                return general_solution
            
            equations = []
            for deriv_order, point, value in ics:
                expr = rhs
                for _ in range(deriv_order):
                    expr = expr.diff(x)
                
                try:
                    eq = expr.subs(x, point) - value
                    equations.append(eq)
                except:
                    pass
            
            if equations:
                try:
                    const_solution = sp.solve(equations, constants, dict=True)
                    
                    const_values = None
                    if isinstance(const_solution, list) and len(const_solution) > 0:
                        if isinstance(const_solution[0], dict):
                            const_values = const_solution[0]
                    elif isinstance(const_solution, dict):
                        const_values = const_solution
                    
                    if const_values and len(const_values) > 0:
                        print(f"✓ Constants determined")
                        particular_solution = rhs.subs(const_values)
                        try:
                            particular_solution = simplify(particular_solution)
                        except:
                            pass
                        return Eq(y(x), particular_solution)
                except:
                    pass
            
            return general_solution
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return general_solution
    
    def _display_ode_solution(self, solution_dict: Dict):
        """Display ODE solution"""
        print("\n" + "=" * 80)
        print("SOLUTION")
        print("=" * 80)
        
        solution = solution_dict['solution']
        
        if isinstance(solution, Eq):
            print(f"\n✅ {solution.lhs} = {solution.rhs}")
            
            rhs = solution.rhs
            x = solution_dict['variable']
            free_syms = rhs.free_symbols - {x}
            
            if free_syms:
                print(f"\n⚠ Undetermined constants: {', '.join(map(str, free_syms))}")
                print("  💡 Provide ICs for particular solution")
        else:
            print(f"\n✅ {solution}")
        
        print("\n" + "=" * 80)
        print("💡 Next: 'visualize' or 'export'")
        print("=" * 80)
    
    # ========================================================================
    # ODE SYSTEM SOLVER
    # ========================================================================
    
    def solve_ode_system(self):
        """Solve system of ODEs"""
        print("\n" + "=" * 80)
        print("SOLVE SYSTEM OF ODEs")
        print("=" * 80)
        
        print("\n📖 Examples:")
        print("  x' = -y")
        print("  y' = x")
        
        n_str = input("\n🔹 Number of equations: ").strip()
        
        is_valid, n = self.validator.validate_number(n_str, allow_negative=False)
        if not is_valid or n < 1:
            print("❌ Invalid number")
            return
        
        n = int(n)
        
        equations = []
        functions = []
        
        print(f"\n📝 Enter {n} equations:")
        
        for i in range(n):
            eq_str = input(f"  Equation {i+1}: ").strip()
            
            if not eq_str:
                print(f"❌ Empty equation")
                return
            
            is_valid, msg = self.validator.validate_equation(eq_str)
            if not is_valid:
                print(f"❌ Invalid: {msg}")
                return
            
            equations.append(eq_str)
            
            if "'" in eq_str:
                func_name = eq_str.split("'")[0].strip()
                if func_name and func_name not in functions:
                    functions.append(func_name)
        
        if len(functions) == 0:
            print("❌ No functions found")
            return
        
        print(f"\n📊 Functions: {', '.join(functions)}")
        
        print("\n📋 Initial conditions (optional):")
        print("  Example: x(0)=1, y(0)=0")
        ics_str = input("  ICs: ").strip()
        
        try:
            solution = self._solve_ode_system_internal(equations, functions, ics_str)
            
            if solution is None:
                return
            
            self.last_solution = solution
            self.last_type = 'ode_system'
            self.last_equations = equations
            
            self._display_ode_system_solution(solution)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if input("Show details? (y/n): ").strip().lower() == 'y':
                traceback.print_exc()
    
    def _solve_ode_system_internal(self, equations: List[str], 
                                   func_names: List[str], 
                                   ics_str: str) -> Optional[Dict]:
        """Internal ODE system solver"""
        try:
            t = symbols('t', real=True)
            funcs = [Function(f)(t) for f in func_names]
            
            print("\n" + "-" * 80)
            print("🔄 Parsing...")
            print("-" * 80)
            
            eqs = []
            for i, eq_str in enumerate(equations):
                eq = self._parse_system_ode(eq_str, func_names, funcs, t)
                if eq is None:
                    return None
                eqs.append(eq)
                print(f"  ✓ Equation {i+1}")
            
            print("\n" + "-" * 80)
            print("🔄 Solving...")
            print("-" * 80)
            
            try:
                solution = dsolve(eqs, funcs)
                
                if not isinstance(solution, list):
                    solution = [solution]
                
                print(f"✓ System solved")
                
            except Exception as e:
                print(f"⚠ Symbolic solution failed: {str(e)}")
                print("💡 Try 'numerical' solver")
                return None
            
            if ics_str:
                solution = self._apply_system_ics(solution, ics_str, func_names, funcs, t)
            
            return {
                'solution': solution,
                'functions': func_names,
                'variable': t,
                'equations': eqs,
                'equation_strs': equations,
                'type': 'ode_system'
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def _parse_system_ode(self, eq_str: str, func_names: List[str], 
                         funcs: List, var) -> Optional[Eq]:
        """Parse system ODE equation"""
        try:
            if '=' not in eq_str:
                print(f"❌ Must contain '=': {eq_str}")
                return None
            
            parts = eq_str.split('=')
            if len(parts) != 2:
                print(f"❌ Must have one '=': {eq_str}")
                return None
            
            left, right = parts
            left = left.strip()
            right = right.strip()
            
            if "'" in left:
                func_name = left.replace("'", "").strip()
                if func_name not in func_names:
                    print(f"❌ Unknown function: {func_name}")
                    return None
                idx = func_names.index(func_name)
                left_expr = funcs[idx].diff(var)
            else:
                if left not in func_names:
                    print(f"❌ Unknown function: {left}")
                    return None
                idx = func_names.index(left)
                left_expr = funcs[idx]
            
            right_expr = self._parse_system_expression(right, func_names, funcs, var)
            
            if right_expr is None:
                return None
            
            return Eq(left_expr, right_expr)
            
        except Exception as e:
            print(f"❌ Parse error: {str(e)}")
            return None
    
    def _parse_system_expression(self, expr_str: str, func_names: List[str], 
                                 funcs: List, var) -> Optional[sp.Basic]:
        """Parse system expression"""
        try:
            parsed = expr_str
            sorted_names = sorted(func_names, key=len, reverse=True)
            
            for name in sorted_names:
                pattern = r'\b' + re.escape(name) + r'\b'
                replacement = f'{name}({var})'
                parsed = re.sub(pattern, replacement, parsed)
            
            namespace = {
                str(var): var, 'exp': exp, 'sin': sin, 'cos': cos,
                'tan': tan, 'log': log, 'sqrt': sqrt, 'pi': pi, 
                'e': E, 'E': E, 'I': I
            }
            
            for i, name in enumerate(func_names):
                namespace[f'{name}({var})'] = funcs[i]
            
            transformations = standard_transformations + (implicit_multiplication_application,)
            result = parse_expr(parsed, local_dict=namespace, transformations=transformations)
            
            return result
            
        except Exception as e:
            print(f"❌ Could not parse '{expr_str}': {str(e)}")
            return None
    
    def _apply_system_ics(self, solution: List[Eq], ics_str: str, 
                         func_names: List[str], funcs: List, var) -> List[Eq]:
        """Apply ICs to system"""
        print("\n" + "=" * 80)
        print("APPLYING INITIAL CONDITIONS")
        print("=" * 80)
        
        try:
            constants = set()
            for sol in solution:
                constants.update(sol.rhs.free_symbols - {var})
            
            constants = sorted(list(constants), key=str)
            
            if not constants:
                print("✓ No constants")
                return solution
            
            print(f"📊 Constants: {', '.join(map(str, constants))}")
            
            ics = {}
            ic_parts = [ic.strip() for ic in ics_str.split(',') if ic.strip()]
            
            for ic_str_part in ic_parts:
                match = re.match(r'(\w+)\(([^)]+)\)\s*=\s*(.+)', ic_str_part.strip())
                if match:
                    func_name = match.group(1)
                    point_str = match.group(2)
                    value_str = match.group(3)
                    
                    try:
                        point = float(sympify(point_str))
                        value = float(sympify(value_str))
                        
                        if func_name in func_names:
                            idx = func_names.index(func_name)
                            ics[funcs[idx]] = (point, value)
                            print(f"  ✓ {func_name}({point}) = {value}")
                    except Exception as e:
                        print(f"  ⚠ Could not parse '{ic_str_part}': {str(e)}")
            
            if not ics:
                print("⚠ No valid ICs")
                return solution
            
            equations = []
            for func, (point, value) in ics.items():
                for sol in solution:
                    if sol.lhs == func:
                        try:
                            eq = sol.rhs.subs(var, point) - value
                            equations.append(eq)
                        except:
                            pass
            
            if equations:
                try:
                    const_values = sp.solve(equations, constants, dict=True)
                    
                    if isinstance(const_values, list) and len(const_values) > 0:
                        if isinstance(const_values[0], dict):
                            const_values = const_values[0]
                    elif not isinstance(const_values, dict):
                        const_values = {}
                    
                    if const_values and len(const_values) > 0:
                        print(f"✓ Constants determined")
                        
                        final_solution = []
                        for sol in solution:
                            new_rhs = sol.rhs.subs(const_values)
                            try:
                                new_rhs = simplify(new_rhs)
                            except:
                                pass
                            final_solution.append(Eq(sol.lhs, new_rhs))
                        
                        return final_solution
                except:
                    pass
            
            return solution
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return solution
    
    def _display_ode_system_solution(self, solution_dict: Dict):
        """Display ODE system solution"""
        print("\n" + "=" * 80)
        print("SYSTEM SOLUTION")
        print("=" * 80)
        
        solution = solution_dict['solution']
        func_names = solution_dict['functions']
        var = solution_dict['variable']
        
        for i, sol in enumerate(solution):
            if i < len(func_names):
                print(f"\n✅ {func_names[i]}({var}) = {sol.rhs}")
                
                free_syms = sol.rhs.free_symbols - {var}
                if free_syms:
                    print(f"   ⚠ Contains: {', '.join(map(str, free_syms))}")
        
        print("\n" + "=" * 80)
        print("💡 Next: 'visualize', 'stability', or 'export'")
        print("=" * 80)
    
    # ========================================================================
    # PDE SOLVER
    # ========================================================================
    
    def solve_pde(self):
        """Solve single PDE"""
        print("\n" + "=" * 80)
        print("SOLVE PDE")
        print("=" * 80)
        
        print("\n📖 Notation:")
        print("  u_t, u_x   → ∂u/∂t, ∂u/∂x")
        print("  u_xx, u_tt → ∂²u/∂x², ∂²u/∂t²")
        
        print("\n💡 Examples:")
        print("  u_t = alpha*u_xx  (Heat)")
        print("  u_tt = c**2*u_xx  (Wave)")
        
        pde_str = input("\n🔹 Enter PDE: ").strip()
        
        if not pde_str:
            print("❌ No equation")
            return
        
        is_valid, msg = self.validator.validate_equation(pde_str)
        if not is_valid:
            print(f"❌ Invalid: {msg}")
            return
        
        try:
            solution = self._solve_pde_internal(pde_str)
            
            if solution is None:
                return
            
            self.last_solution = solution
            self.last_type = 'pde'
            self.last_equations = pde_str
            
            self._display_pde_solution(solution)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if input("Show details? (y/n): ").strip().lower() == 'y':
                traceback.print_exc()
    
    def _solve_pde_internal(self, pde_str: str) -> Optional[Dict]:
        """Internal PDE solver"""
        try:
            x, t = symbols('x t', real=True)
            u = Function('u')
            
            pde_eq = self._parse_pde(pde_str, x, t, u)
            if pde_eq is None:
                return None
            
            print("\n" + "-" * 80)
            print(f"📋 Equation: {pde_eq}")
            print("🔄 Solving...")
            print("-" * 80)
            
            try:
                from sympy import pdsolve
                solution = pdsolve(pde_eq, u(x, t))
                print(f"✓ Symbolic solution found")
                
                return {
                    'solution': solution,
                    'variables': (x, t),
                    'function': u,
                    'equation': pde_eq,
                    'equation_str': pde_str,
                    'type': 'pde'
                }
                
            except Exception as e:
                print(f"⚠ Symbolic solution not available: {str(e)}")
                print("💡 Use 'numerical' for visualization")
                
                return {
                    'solution': pde_eq,
                    'variables': (x, t),
                    'function': u,
                    'equation': pde_eq,
                    'equation_str': pde_str,
                    'numerical': True,
                    'type': 'pde'
                }
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def _parse_pde(self, pde_str: str, x, t, u) -> Optional[Eq]:
        """Parse PDE string"""
        try:
            pde_str = pde_str.replace("u_tt", "Derivative(u(x,t), t, t)")
            pde_str = pde_str.replace("u_xx", "Derivative(u(x,t), x, x)")
            pde_str = pde_str.replace("u_xt", "Derivative(u(x,t), x, t)")
            pde_str = pde_str.replace("u_tx", "Derivative(u(x,t), t, x)")
            pde_str = pde_str.replace("u_t", "Derivative(u(x,t), t)")
            pde_str = pde_str.replace("u_x", "Derivative(u(x,t), x)")
            pde_str = re.sub(r'\bu\b(?!\()', 'u(x,t)', pde_str)
            
            if '=' in pde_str:
                parts = pde_str.split('=')
                if len(parts) != 2:
                    print("❌ Must have one '='")
                    return None
                left, right = parts
            else:
                left = pde_str
                right = '0'
            
            namespace = {
                'x': x, 't': t, 'u': u, 'exp': exp, 'sin': sin, 'cos': cos,
                'tan': tan, 'log': log, 'sqrt': sqrt, 'pi': pi, 'e': E, 'E': E,
                'Derivative': Derivative,
                'alpha': symbols('alpha', positive=True, real=True),
                'beta': symbols('beta', real=True),
                'gamma': symbols('gamma', real=True),
                'c': symbols('c', positive=True, real=True),
                'D': symbols('D', positive=True, real=True)
            }
            
            try:
                transformations = standard_transformations + (implicit_multiplication_application,)
                left_expr = parse_expr(left.strip(), local_dict=namespace, 
                                      transformations=transformations)
                right_expr = parse_expr(right.strip(), local_dict=namespace, 
                                       transformations=transformations)
            except Exception as e:
                print(f"❌ Parse error: {str(e)}")
                return None
            
            return Eq(left_expr, right_expr)
            
        except Exception as e:
            print(f"❌ Could not parse: {str(e)}")
            return None
    
    def _display_pde_solution(self, solution_dict: Dict):
        """Display PDE solution"""
        print("\n" + "=" * 80)
        print("SOLUTION")
        print("=" * 80)
        
        solution = solution_dict['solution']
        
        if isinstance(solution, Eq) and solution.lhs != solution.rhs:
            print(f"\n✅ {solution}")
        else:
            print(f"\n📋 Equation: {solution}")
            print("⚠ Analytical solution not available")
            print("💡 Use 'numerical' for visualization")
        
        print("\n" + "=" * 80)
        print("💡 Next: 'numerical' or 'export'")
        print("=" * 80)
    
    # ========================================================================
    # PDE SYSTEM SOLVER
    # ========================================================================
    
    def solve_pde_system(self):
        """Solve system of PDEs"""
        print("\n" + "=" * 80)
        print("SOLVE SYSTEM OF PDEs")
        print("=" * 80)
        
        print("\n📖 Examples:")
        print("  u_t = D1*u_xx + alpha*u - beta*u*v")
        print("  v_t = D2*v_xx + beta*u*v - gamma*v")
        
        n_str = input("\n🔹 Number of equations: ").strip()
        
        is_valid, n = self.validator.validate_number(n_str, allow_negative=False)
        if not is_valid or n < 1:
            print("❌ Invalid number")
            return
        
        n = int(n)
        
        equations = []
        functions = []
        
        print(f"\n📝 Enter {n} equations:")
        
        for i in range(n):
            eq_str = input(f"  Equation {i+1}: ").strip()
            
            if not eq_str:
                print(f"❌ Empty equation")
                return
            
            is_valid, msg = self.validator.validate_equation(eq_str)
            if not is_valid:
                print(f"❌ Invalid: {msg}")
                return
            
            equations.append(eq_str)
            
            if "_" in eq_str:
                func_name = eq_str.split("_")[0].strip()
                if func_name and func_name not in functions:
                    functions.append(func_name)
        
        if len(functions) == 0:
            print("❌ No functions found")
            return
        
        print(f"\n📊 Functions: {', '.join(functions)}")
        
        try:
            solution = self._solve_pde_system_internal(equations, functions)
            
            if solution is None:
                return
            
            self.last_solution = solution
            self.last_type = 'pde_system'
            self.last_equations = equations
            
            self._display_pde_system_solution(solution)
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if input("Show details? (y/n): ").strip().lower() == 'y':
                traceback.print_exc()
    
    def _solve_pde_system_internal(self, equations: List[str], 
                                   func_names: List[str]) -> Optional[Dict]:
        """Internal PDE system solver"""
        try:
            x, t = symbols('x t', real=True)
            funcs = [Function(f)(x, t) for f in func_names]
            
            print("\n" + "-" * 80)
            print("🔄 Parsing...")
            print("-" * 80)
            
            eqs = []
            for i, eq_str in enumerate(equations):
                eq = self._parse_system_pde(eq_str, func_names, funcs, x, t)
                if eq is None:
                    return None
                eqs.append(eq)
                print(f"  ✓ Equation {i+1}")
            
            print("\n⚠ Analytical solutions for PDE systems rarely available")
            print("  Use 'numerical' for reaction-diffusion systems")
            
            return {
                'equations': eqs,
                'functions': func_names,
                'variables': (x, t),
                'symbolic_funcs': funcs,
                'equation_strs': equations,
                'type': 'pde_system'
            }
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return None
    
    def _parse_system_pde(self, eq_str: str, func_names: List[str], 
                         funcs: List, x, t) -> Optional[Eq]:
        """Parse PDE system equation"""
        try:
            for i, name in enumerate(func_names):
                eq_str = eq_str.replace(f"{name}_tt", f"Derivative({name}(x,t), t, t)")
                eq_str = eq_str.replace(f"{name}_xx", f"Derivative({name}(x,t), x, x)")
                eq_str = eq_str.replace(f"{name}_xt", f"Derivative({name}(x,t), x, t)")
                eq_str = eq_str.replace(f"{name}_t", f"Derivative({name}(x,t), t)")
                eq_str = eq_str.replace(f"{name}_x", f"Derivative({name}(x,t), x)")
                pattern = r'\b' + re.escape(name) + r'\b(?!\()'
                eq_str = re.sub(pattern, f'{name}(x,t)', eq_str)
            
            if '=' not in eq_str:
                print(f"❌ Must contain '=': {eq_str}")
                return None
            
            parts = eq_str.split('=')
            if len(parts) != 2:
                print(f"❌ Must have one '=': {eq_str}")
                return None
            
            left, right = parts
            
            namespace = {
                'x': x, 't': t, 'exp': exp, 'sin': sin, 'cos': cos,
                'Derivative': Derivative,
                'alpha': symbols('alpha', real=True),
                'beta': symbols('beta', real=True),
                'gamma': symbols('gamma', real=True),
                'delta': symbols('delta', real=True),
                'D1': symbols('D1', positive=True, real=True),
                'D2': symbols('D2', positive=True, real=True),
                'c': symbols('c', positive=True, real=True)
            }
            
            for i, name in enumerate(func_names):
                namespace[f'{name}(x,t)'] = funcs[i]
            
            try:
                transformations = standard_transformations + (implicit_multiplication_application,)
                left_expr = parse_expr(left.strip(), local_dict=namespace, 
                                      transformations=transformations)
                right_expr = parse_expr(right.strip(), local_dict=namespace, 
                                       transformations=transformations)
            except Exception as e:
                print(f"❌ Parse error: {str(e)}")
                return None
            
            return Eq(left_expr, right_expr)
            
        except Exception as e:
            print(f"❌ Could not parse: {str(e)}")
            return None
    
    def _display_pde_system_solution(self, solution_dict: Dict):
        """Display PDE system"""
        print("\n" + "=" * 80)
        print("SYSTEM OF PDEs")
        print("=" * 80)
        
        equations = solution_dict['equations']
        func_names = solution_dict['functions']
        
        print("\n📋 Symbolic form:")
        for i, eq in enumerate(equations):
            if i < len(func_names):
                print(f"\n  {func_names[i]}: {eq}")
        
        print("\n" + "=" * 80)
        print("💡 Use 'numerical' for reaction-diffusion visualization")
        print("=" * 80)
    
    # ========================================================================
    # NUMERICAL SOLVER
    # ========================================================================
    
    def solve_numerical(self):
        """Numerical solver interface"""
        print("\n" + "=" * 80)
        print("NUMERICAL SOLVER")
        print("=" * 80)
        
        print("\n📊 Select type:")
        print("  1 - Heat Equation")
        print("  2 - Wave Equation")
        print("  3 - Reaction-Diffusion System")
        print("  4 - Custom ODE System")
        
        choice = input("\n🔹 Choice (1-4): ").strip()
        
        if choice == '1':
            self._solve_heat_numerical()
        elif choice == '2':
            self._solve_wave_numerical()
        elif choice == '3':
            self._solve_reaction_diffusion_numerical()
        elif choice == '4':
            self._solve_custom_ode_numerical()
        else:
            print("❌ Invalid choice")
    
    def _solve_heat_numerical(self):
        """Solve heat equation numerically"""
        print("\n" + "-" * 80)
        print("HEAT EQUATION: u_t = α·u_xx")
        print("-" * 80)
        
        print("\n📋 Domain: 0 ≤ x ≤ L, 0 ≤ t ≤ T")
        print("  BC: u(0,t) = u(L,t) = 0")
        print("  IC: u(x,0) = sin(πx/L)")
        
        alpha_str = input("\n🔹 Diffusion α (default=0.01): ").strip()
        alpha = float(alpha_str) if alpha_str else 0.01
        
        L_str = input("🔹 Length L (default=1.0): ").strip()
        L = float(L_str) if L_str else 1.0
        
        T_str = input("🔹 Time T (default=1.0): ").strip()
        T = float(T_str) if T_str else 1.0
        
        print("\n🔄 Solving...")
        result = self.numerical_solver.solve_heat_equation_numerical(
            alpha=alpha, L=L, T=T, nx=100, nt=200
        )
        
        if result['success']:
            print(f"✓ Solution computed (r={result['r']:.4f})")
            
            self.last_solution = {
                'type': 'numerical_pde',
                'x': result['x'],
                't': result['t'],
                'u': result['u'],
                'equation': f"Heat Eq. (α={alpha})",
                'pde_type': 'heat',
                'params': {'alpha': alpha, 'L': L, 'T': T}
            }
            self.last_type = 'numerical_pde'
            
            self._visualize_numerical_pde(result, f"Heat Equation (α={alpha})")
        else:
            print(f"❌ Failed: {result['message']}")
    
    def _solve_wave_numerical(self):
        """Solve wave equation numerically"""
        print("\n" + "-" * 80)
        print("WAVE EQUATION: u_tt = c²·u_xx")
        print("-" * 80)
        
        print("\n📋 Domain: 0 ≤ x ≤ L, 0 ≤ t ≤ T")
        print("  BC: u(0,t) = u(L,t) = 0")
        print("  IC: u(x,0) = sin(πx/L), u_t(x,0) = 0")
        
        c_str = input("\n🔹 Wave speed c (default=1.0): ").strip()
        c = float(c_str) if c_str else 1.0
        
        L_str = input("🔹 Length L (default=1.0): ").strip()
        L = float(L_str) if L_str else 1.0
        
        T_str = input("🔹 Time T (default=2.0): ").strip()
        T = float(T_str) if T_str else 2.0
        
        print("\n🔄 Solving...")
        result = self.numerical_solver.solve_wave_equation_numerical(
            c=c, L=L, T=T, nx=100, nt=200
        )
        
        if result['success']:
            print(f"✓ Solution computed (CFL={result['r']:.4f})")
            
            self.last_solution = {
                'type': 'numerical_pde',
                'x': result['x'],
                't': result['t'],
                'u': result['u'],
                'equation': f"Wave Eq. (c={c})",
                'pde_type': 'wave',
                'params': {'c': c, 'L': L, 'T': T}
            }
            self.last_type = 'numerical_pde'
            
            self._visualize_numerical_pde(result, f"Wave Equation (c={c})")
        else:
            print(f"❌ Failed: {result['message']}")
    
    def _solve_reaction_diffusion_numerical(self):
        """Solve reaction-diffusion system"""
        print("\n" + "-" * 80)
        print("REACTION-DIFFUSION SYSTEM")
        print("-" * 80)
        
        print("\n📋 System:")
        print("  u_t = D1·u_xx + α·u - β·u·v")
        print("  v_t = D2·v_xx + β·u·v - α·v")
        
        D1_str = input("\n🔹 D1 (default=0.1): ").strip()
        D1 = float(D1_str) if D1_str else 0.1
        
        D2_str = input("🔹 D2 (default=0.05): ").strip()
        D2 = float(D2_str) if D2_str else 0.05
        
        alpha_str = input("🔹 α (default=1.0): ").strip()
        alpha = float(alpha_str) if alpha_str else 1.0
        
        beta_str = input("🔹 β (default=0.5): ").strip()
        beta = float(beta_str) if beta_str else 0.5
        
        print("\n🔄 Solving...")
        result = self.numerical_solver.solve_reaction_diffusion_numerical(
            D1=D1, D2=D2, alpha=alpha, beta=beta, L=10.0, T=20.0, nx=100, nt=200
        )
        
        if result['success']:
            print(f"✓ Solution computed")
            
            self.last_solution = {
                'type': 'numerical_pde_system',
                'x': result['x'],
                't': result['t'],
                'u': result['u'],
                'v': result['v'],
                'equation': f"Reaction-Diffusion",
                'params': {'D1': D1, 'D2': D2, 'alpha': alpha, 'beta': beta}
            }
            self.last_type = 'numerical_pde_system'
            
            self._visualize_reaction_diffusion(result)
        else:
            print(f"❌ Failed: {result['message']}")
    
    def _solve_custom_ode_numerical(self):
        """Solve custom ODE system numerically"""
        print("\n" + "-" * 80)
        print("CUSTOM ODE SYSTEM (Predator-Prey Example)")
        print("-" * 80)
        
        print("\n📋 Lotka-Volterra equations:")
        print("  x' = α·x - β·x·y  (prey)")
        print("  y' = δ·x·y - γ·y  (predator)")
        
        alpha = 1.0
        beta = 0.1
        delta = 0.075
        gamma = 1.5
        
        def lotka_volterra(t, y):
            x, y_val = y
            dx = alpha * x - beta * x * y_val
            dy = delta * x * y_val - gamma * y_val
            return [dx, dy]
        
        y0 = [10, 5]
        t_span = (0, 50)
        t_eval = np.linspace(0, 50, 1000)
        
        print("\n🔄 Solving...")
        result = self.numerical_solver.solve_ode_numerical(
            lotka_volterra, y0, t_span, t_eval
        )
        
        if result['success']:
            print("✓ Solution computed")
            
            self.last_solution = {
                'type': 'numerical_ode_system',
                't': result['t'],
                'y': result['y'],
                'functions': ['prey', 'predator'],
                'equation': 'Lotka-Volterra'
            }
            self.last_type = 'numerical_ode_system'
            
            self._visualize_numerical_ode_system(result)
        else:
            print(f"❌ Failed: {result['message']}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def visualize(self):
        """Visualize last solution"""
        if self.last_solution is None:
            print("\n❌ No solution to visualize")
            return
        
        print("\n" + "=" * 80)
        print("VISUALIZING SOLUTION")
        print("=" * 80)
        
        try:
            if self.last_type == 'ode':
                self._visualize_ode(self.last_solution)
            elif self.last_type == 'ode_system':
                self._visualize_ode_system(self.last_solution)
            elif self.last_type == 'numerical_pde':
                pde_type = self.last_solution.get('pde_type', 'unknown')
                title = self.last_solution.get('equation', 'PDE')
                self._visualize_numerical_pde(self.last_solution, title)
            elif self.last_type == 'numerical_pde_system':
                self._visualize_reaction_diffusion(self.last_solution)
            elif self.last_type == 'numerical_ode_system':
                self._visualize_numerical_ode_system(self.last_solution)
            else:
                print(f"\n⚠ Visualization not available for: {self.last_type}")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if input("Show details? (y/n): ").strip().lower() == 'y':
                traceback.print_exc()
    
    def _visualize_ode(self, solution_dict: Dict):
        """Visualize ODE solution"""
        from sympy import lambdify
        
        solution = solution_dict['solution']
        var = solution_dict['variable']
        eq_str = solution_dict.get('equation_str', 'ODE')
        
        if isinstance(solution, Eq):
            expr = solution.rhs
        else:
            expr = solution
        
        free_symbols = expr.free_symbols - {var}
        if free_symbols:
            print(f"\n⚠ Undetermined constants: {', '.join(map(str, free_symbols))}")
            print("  Cannot visualize")
            return
        
        try:
            func = lambdify(var, expr, modules=['numpy'])
        except Exception as e:
            print(f"❌ Could not create function: {str(e)}")
            return
        
        x_vals = np.linspace(0, 10, 1000)
        
        try:
            y_vals = func(x_vals)
            
            if np.iscomplexobj(y_vals):
                print("⚠ Complex-valued, plotting real part")
                y_vals = np.real(y_vals)
            
            if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
                print("⚠ Filtering invalid values")
                valid_mask = np.isfinite(y_vals)
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]
                
                if len(x_vals) == 0:
                    print("❌ No valid points")
                    return
        except Exception as e:
            print(f"❌ Error evaluating: {str(e)}")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'ODE Solution: {eq_str}', fontsize=16, fontweight='bold')
        
        # Solution
        axes[0].plot(x_vals, y_vals, 'b-', linewidth=2.5, label='y(x)')
        axes[0].set_xlabel('x', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('y(x)', fontsize=12, fontweight='bold')
        axes[0].set_title('Solution', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        axes[0].axvline(x=0, color='k', linewidth=0.5)
        axes[0].legend()
        
        # First derivative
        try:
            expr_prime = expr.diff(var)
            func_prime = lambdify(var, expr_prime, modules=['numpy'])
            y_prime_vals = func_prime(x_vals)
            
            if np.iscomplexobj(y_prime_vals):
                y_prime_vals = np.real(y_prime_vals)
            
            axes[1].plot(x_vals, y_prime_vals, 'r-', linewidth=2.5, label="y'(x)")
            axes[1].set_xlabel('x', fontsize=12, fontweight='bold')
            axes[1].set_ylabel("y'(x)", fontsize=12, fontweight='bold')
            axes[1].set_title("First Derivative", fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].axhline(y=0, color='k', linewidth=0.5)
            axes[1].axvline(x=0, color='k', linewidth=0.5)
            axes[1].legend()
        except:
            axes[1].text(0.5, 0.5, 'Could not compute derivative', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        # Second derivative
        try:
            expr_double_prime = expr_prime.diff(var)
            func_double_prime = lambdify(var, expr_double_prime, modules=['numpy'])
            y_double_prime_vals = func_double_prime(x_vals)
            
            if np.iscomplexobj(y_double_prime_vals):
                y_double_prime_vals = np.real(y_double_prime_vals)
            
            axes[2].plot(x_vals, y_double_prime_vals, 'g-', linewidth=2.5, label="y''(x)")
            axes[2].set_xlabel('x', fontsize=12, fontweight='bold')
            axes[2].set_ylabel("y''(x)", fontsize=12, fontweight='bold')
            axes[2].set_title("Second Derivative", fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3, linestyle='--')
            axes[2].axhline(y=0, color='k', linewidth=0.5)
            axes[2].axvline(x=0, color='k', linewidth=0.5)
            axes[2].legend()
        except:
            axes[2].text(0.5, 0.5, 'Could not compute second derivative', 
                        ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        self.save_plot(fig, "ode_solution")
        print("✓ Visualization complete")
    
    def _visualize_ode_system(self, solution_dict: Dict):
        """Visualize ODE system"""
        from sympy import lambdify
        
        solution = solution_dict['solution']
        func_names = solution_dict['functions']
        var = solution_dict['variable']
        eq_strs = solution_dict.get('equation_strs', [])
        
        for sol in solution:
            free_symbols = sol.rhs.free_symbols - {var}
            if free_symbols:
                print(f"\n⚠ Undetermined constants: {', '.join(map(str, free_symbols))}")
                print("  💡 Provide ICs")
                return
        
        funcs = []
        for sol in solution:
            try:
                f = lambdify(var, sol.rhs, modules=['numpy'])
                funcs.append(f)
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                return
        
        t_vals = np.linspace(0, 10, 1000)
        
        n_funcs = len(funcs)
        n_plots = n_funcs + (1 if n_funcs == 2 else 0) + (1 if n_funcs >= 3 else 0)
        
        fig = plt.figure(figsize=(14, 5 * n_plots))
        
        system_title = "System of ODEs"
        if eq_strs:
            system_title += f": {', '.join(eq_strs[:2])}"
        fig.suptitle(system_title, fontsize=16, fontweight='bold')
        
        all_vals = []
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for i, (func, name) in enumerate(zip(funcs, func_names)):
            try:
                vals = func(t_vals)
                
                if np.iscomplexobj(vals):
                    vals = np.real(vals)
                
                if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                    valid_mask = np.isfinite(vals)
                    vals = np.where(valid_mask, vals, 0)
                
                all_vals.append(vals)
                
                ax = plt.subplot(n_plots, 1, i + 1)
                color = colors[i % len(colors)]
                ax.plot(t_vals, vals, color=color, linewidth=2.5, label=f'{name}(t)')
                ax.set_xlabel('t', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{name}(t)', fontsize=12, fontweight='bold')
                ax.set_title(f'Solution: {name}(t)', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.legend()
            except Exception as e:
                print(f"⚠ Error plotting {name}: {str(e)}")
        
        # 2D phase portrait
        if n_funcs == 2 and len(all_vals) == 2:
            try:
                ax = plt.subplot(n_plots, 1, n_funcs + 1)
                ax.plot(all_vals[0], all_vals[1], 'purple', linewidth=2.5, label='Trajectory')
                ax.plot(all_vals[0][0], all_vals[1][0], 'go', markersize=12, label='Start', zorder=5)
                ax.plot(all_vals[0][-1], all_vals[1][-1], 'ro', markersize=12, label='End', zorder=5)
                ax.set_xlabel(f'{func_names[0]}(t)', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{func_names[1]}(t)', fontsize=12, fontweight='bold')
                ax.set_title('2D Phase Portrait', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.legend()
            except:
                pass
        
        # 3D phase portrait
        if n_funcs >= 3 and len(all_vals) >= 3:
            try:
                ax = fig.add_subplot(n_plots, 1, n_plots, projection='3d')
                ax.plot(all_vals[0], all_vals[1], all_vals[2], 
                       'purple', linewidth=2, label='Trajectory')
                ax.scatter(all_vals[0][0], all_vals[1][0], all_vals[2][0], 
                          color='green', s=100, label='Start', zorder=5)
                ax.scatter(all_vals[0][-1], all_vals[1][-1], all_vals[2][-1], 
                          color='red', s=100, label='End', zorder=5)
                ax.set_xlabel(f'{func_names[0]}(t)', fontsize=10, fontweight='bold')
                ax.set_ylabel(f'{func_names[1]}(t)', fontsize=10, fontweight='bold')
                ax.set_zlabel(f'{func_names[2]}(t)', fontsize=10, fontweight='bold')
                ax.set_title('3D Phase Portrait', fontsize=14, fontweight='bold')
                ax.legend()
            except:
                pass
        
        plt.tight_layout()
        self.save_plot(fig, "ode_system_solution")
        print("✓ Visualization complete")
    
    def _visualize_numerical_pde(self, result: Dict, title: str):
        """Visualize numerical PDE solution"""
        x = result['x']
        t = result['t']
        u = result['u']
        
        X, T = np.meshgrid(x, t)
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'PDE Solution: {title}', fontsize=16, fontweight='bold')
        
        # 3D Surface
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf = ax1.plot_surface(X, T, u, cmap=cm.viridis, 
                               linewidth=0, antialiased=True, alpha=0.9)
        ax1.set_xlabel('x', fontsize=12, fontweight='bold')
        ax1.set_ylabel('t', fontsize=12, fontweight='bold')
        ax1.set_zlabel('u(x,t)', fontsize=12, fontweight='bold')
        ax1.set_title('3D Surface', fontsize=14, fontweight='bold')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
        
        # Heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        im = ax2.imshow(u, extent=[X.min(), X.max(), T.min(), T.max()], 
                       origin='lower', cmap='hot', aspect='auto', interpolation='bilinear')
        ax2.set_xlabel('x', fontsize=12, fontweight='bold')
        ax2.set_ylabel('t', fontsize=12, fontweight='bold')
        ax2.set_title('Heatmap', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax2)
        
        # Contour
        ax3 = fig.add_subplot(2, 2, 3)
        contour = ax3.contour(X, T, u, levels=20, cmap='coolwarm')
        ax3.clabel(contour, inline=True, fontsize=8)
        ax3.set_xlabel('x', fontsize=12, fontweight='bold')
        ax3.set_ylabel('t', fontsize=12, fontweight='bold')
        ax3.set_title('Contour Plot', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Snapshots
        ax4 = fig.add_subplot(2, 2, 4)
        n_snapshots = 5
        time_indices = np.linspace(0, len(T) - 1, n_snapshots, dtype=int)
        colors_snap = ['blue', 'green', 'orange', 'red', 'purple']
        
        for idx, color in zip(time_indices, colors_snap):
            t_val = T[idx, 0]
            ax4.plot(X[idx, :], u[idx, :], color=color, linewidth=2, 
                    label=f't = {t_val:.3f}')
        
        ax4.set_xlabel('x', fontsize=12, fontweight='bold')
        ax4.set_ylabel('u(x,t)', fontsize=12, fontweight='bold')
        ax4.set_title('Snapshots', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot(fig, "pde_solution")
        print("✓ Visualization complete")
    
    def _visualize_reaction_diffusion(self, result: Dict):
        """Visualize reaction-diffusion system"""
        x = result['x']
        t = result['t']
        u = result['u']
        v = result['v']
        
        X, T = np.meshgrid(x, t)
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Reaction-Diffusion System', fontsize=16, fontweight='bold')
        
        # u surface
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(X, T, u, cmap=cm.viridis, linewidth=0, alpha=0.9)
        ax1.set_title('u(x,t) Surface', fontsize=14, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u')
        
        # v surface
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        surf2 = ax2.plot_surface(X, T, v, cmap=cm.plasma, linewidth=0, alpha=0.9)
        ax2.set_title('v(x,t) Surface', fontsize=14, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('v')
        
        # u heatmap
        ax3 = fig.add_subplot(2, 3, 3)
        im1 = ax3.imshow(u, extent=[x.min(), x.max(), t.min(), t.max()], 
                        origin='lower', cmap='hot', aspect='auto')
        ax3.set_title('u(x,t) Heatmap', fontsize=14, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        fig.colorbar(im1, ax=ax3)
        
        # v heatmap
        ax4 = fig.add_subplot(2, 3, 4)
        im2 = ax4.imshow(v, extent=[x.min(), x.max(), t.min(), t.max()], 
                        origin='lower', cmap='cool', aspect='auto')
        ax4.set_title('v(x,t) Heatmap', fontsize=14, fontweight='bold')
        ax4.set_xlabel('x')
        ax4.set_ylabel('t')
        fig.colorbar(im2, ax=ax4)
        
        # Final state
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(x, u[-1, :], 'b-', linewidth=2, label='u(x, T)')
        ax5.plot(x, v[-1, :], 'r-', linewidth=2, label='v(x, T)')
        ax5.set_title('Final State', fontsize=14, fontweight='bold')
        ax5.set_xlabel('x')
        ax5.set_ylabel('Concentration')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Phase space
        ax6 = fig.add_subplot(2, 3, 6)
        for i in range(0, len(x), len(x)//10):
            ax6.plot(u[:, i], v[:, i], alpha=0.5, linewidth=1)
        ax6.set_title('Phase Space (u vs v)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('u')
        ax6.set_ylabel('v')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_plot(fig, "reaction_diffusion")
        print("✓ Visualization complete")
    
    def _visualize_numerical_ode_system(self, result: Dict):
        """Visualize numerical ODE system"""
        t = result['t']
        y = result['y']
        func_names = result.get('functions', [f'y{i}' for i in range(len(y))])
        
        n_funcs = len(y)
        n_plots = n_funcs + (1 if n_funcs == 2 else 0)
        
        fig = plt.figure(figsize=(14, 5 * n_plots))
        fig.suptitle('Numerical ODE System', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        
        for i in range(n_funcs):
            ax = plt.subplot(n_plots, 1, i + 1)
            ax.plot(t, y[i], color=colors[i % len(colors)], linewidth=2.5, 
                   label=func_names[i])
            ax.set_xlabel('t', fontsize=12, fontweight='bold')
            ax.set_ylabel(func_names[i], fontsize=12, fontweight='bold')
            ax.set_title(f'Solution: {func_names[i]}(t)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend()
        
        if n_funcs == 2:
            ax = plt.subplot(n_plots, 1, n_plots)
            ax.plot(y[0], y[1], 'purple', linewidth=2.5, label='Trajectory')
            ax.plot(y[0][0], y[1][0], 'go', markersize=12, label='Start', zorder=5)
            ax.plot(y[0][-1], y[1][-1], 'ro', markersize=12, label='End', zorder=5)
            ax.set_xlabel(func_names[0], fontsize=12, fontweight='bold')
            ax.set_ylabel(func_names[1], fontsize=12, fontweight='bold')
            ax.set_title('Phase Portrait', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend()
        
        plt.tight_layout()
        self.save_plot(fig, "numerical_ode_system")
        print("✓ Visualization complete")
    
    # ========================================================================
    # ANIMATION
    # ========================================================================
    
    def create_animation(self):
        """Create animation of solution"""
        if self.last_solution is None:
            print("\n❌ No solution to animate")
            return
        
        print("\n" + "=" * 80)
        print("CREATE ANIMATION")
        print("=" * 80)
        
        if self.last_type not in ['numerical_pde', 'numerical_pde_system']:
            print("\n⚠ Animation only available for numerical PDE solutions")
            print("  Solve a PDE using 'numerical' first")
            return
        
        try:
            if self.last_type == 'numerical_pde':
                self._create_pde_animation(self.last_solution)
            elif self.last_type == 'numerical_pde_system':
                self._create_reaction_diffusion_animation(self.last_solution)
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            if input("Show details? (y/n): ").strip().lower() == 'y':
                traceback.print_exc()
    
    def _create_pde_animation(self, solution_dict: Dict):
        """Create PDE animation"""
        print("\n🎬 Creating animation...")
        
        x = solution_dict['x']
        t = solution_dict['t']
        u = solution_dict['u']
        title = solution_dict.get('equation', 'PDE')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        line, = ax.plot([], [], 'b-', linewidth=2.5)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(u.min() - 0.1 * abs(u.min()), u.max() + 0.1 * abs(u.max()))
        ax.set_xlabel('x', fontsize=12, fontweight='bold')
        ax.set_ylabel('u(x,t)', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} - Evolution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           fontsize=12, verticalalignment='top')
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        def animate(frame):
            line.set_data(x, u[frame, :])
            time_text.set_text(f't = {t[frame]:.3f}')
            return line, time_text
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(t), interval=50, blit=True)
        
        filename = self.generate_filename("pde_animation", "gif")
        
        try:
            writer = PillowWriter(fps=20)
            anim.save(filename, writer=writer)
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"\n✅ Animation saved:")
                print(f"   📄 {os.path.abspath(filename)}")
                print(f"   📊 {file_size / 1024:.2f} KB")
        except Exception as e:
            print(f"\n❌ Could not save animation: {str(e)}")
        finally:
            plt.close(fig)
    
    def _create_reaction_diffusion_animation(self, solution_dict: Dict):
        """Create reaction-diffusion animation"""
        print("\n🎬 Creating animation...")
        
        x = solution_dict['x']
        t = solution_dict['t']
        u = solution_dict['u']
        v = solution_dict['v']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        line_u, = ax1.plot([], [], 'b-', linewidth=2.5, label='u(x,t)')
        line_v, = ax2.plot([], [], 'r-', linewidth=2.5, label='v(x,t)')
        
        ax1.set_xlim(x.min(), x.max())
        ax1.set_ylim(u.min() - 0.1, u.max() + 0.1)
        ax1.set_ylabel('u', fontsize=12, fontweight='bold')
        ax1.set_title('Reaction-Diffusion System', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(v.min() - 0.1, v.max() + 0.1)
        ax2.set_xlabel('x', fontsize=12, fontweight='bold')
        ax2.set_ylabel('v', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                            fontsize=12, verticalalignment='top')
        
        def init():
            line_u.set_data([], [])
            line_v.set_data([], [])
            time_text.set_text('')
            return line_u, line_v, time_text
        
        def animate(frame):
            line_u.set_data(x, u[frame, :])
            line_v.set_data(x, v[frame, :])
            time_text.set_text(f't = {t[frame]:.3f}')
            return line_u, line_v, time_text
        
        anim = FuncAnimation(fig, animate, init_func=init, 
                           frames=len(t), interval=50, blit=True)
        
        filename = self.generate_filename("reaction_diffusion_animation", "gif")
        
        try:
            writer = PillowWriter(fps=20)
            anim.save(filename, writer=writer)
            
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                print(f"\n✅ Animation saved:")
                print(f"   📄 {os.path.abspath(filename)}")
                print(f"   📊 {file_size / 1024:.2f} KB")
        except Exception as e:
            print(f"\n❌ Could not save animation: {str(e)}")
        finally:
            plt.close(fig)
    
    # ========================================================================
    # STABILITY ANALYSIS
    # ========================================================================
    
    def analyze_stability(self):
        """Analyze stability of ODE system"""
        if self.last_solution is None:
            print("\n❌ No solution to analyze")
            return
        
        if self.last_type != 'numerical_ode_system':
            print("\n⚠ Stability analysis only for numerical ODE systems")
            print("  Use 'numerical' → choice 4 first")
            return
        
        print("\n" + "=" * 80)
        print("STABILITY ANALYSIS")
        print("=" * 80)
        
        print("\n⚠ Analysis for Lotka-Volterra system")
        
        # Define system
        alpha, beta, delta, gamma = 1.0, 0.1, 0.075, 1.5
        
        def system(t, y):
            x, y_val = y
            dx = alpha * x - beta * x * y_val
            dy = delta * x * y_val - gamma * y_val
            return [dx, dy]
        
        result = self.stability_analyzer.analyze_ode_equilibria(
            system, ['x', 'y'], search_range=(0, 20)
        )
        
        if result['success']:
            print(f"\n✓ Found {result['count']} equilibrium point(s)")
            
            for i, eq_data in enumerate(result['equilibria']):
                print(f"\n{'='*60}")
                print(f"Equilibrium {i+1}:")
                print(f"  Point: x = {eq_data['point'][0]:.4f}, y = {eq_data['point'][1]:.4f}")
                print(f"  Eigenvalues:")
                for j, ev in enumerate(eq_data['eigenvalues']):
                    if np.isreal(ev):
                        print(f"    λ{j+1} = {np.real(ev):.4f}")
                    else:
                        print(f"    λ{j+1} = {np.real(ev):.4f} + {np.imag(ev):.4f}i")
                print(f"  Stability: {eq_data['stability']}")
        else:
            print(f"\n❌ Analysis failed: {result['message']}")
    
    # ========================================================================
    # EXPORT
    # ========================================================================
    
    def export_solution(self):
        """Export solution"""
        if self.last_solution is None:
            print("\n❌ No solution to export")
            return
        
        print("\n" + "=" * 80)
        print("EXPORT SOLUTION")
        print("=" * 80)
        
        print("\n📁 Formats:")
        print("  1 - JSON")
        print("  2 - LaTeX")
        print("  3 - CSV (numerical data)")
        
        choice = input("\n🔹 Choice (1-3): ").strip()
        
        if choice == '1':
            filename = self.generate_filename("solution", "json")
            if self.exporter.export_to_json(self.last_solution, filename):
                print(f"✅ Exported: {os.path.abspath(filename)}")
        elif choice == '2':
            filename = self.generate_filename("solution", "tex")
            if self.exporter.export_to_latex(self.last_solution, filename):
                print(f"✅ Exported: {os.path.abspath(filename)}")
        elif choice == '3':
            if self.last_type.startswith('numerical'):
                self._export_numerical_data()
            else:
                print("⚠ CSV export only for numerical solutions")
        else:
            print("❌ Invalid choice")
    
    def _export_numerical_data(self):
        """Export numerical data to CSV"""
        try:
            if 't' in self.last_solution and 'y' in self.last_solution:
                # ODE system
                t = self.last_solution['t']
                y = self.last_solution['y']
                func_names = self.last_solution.get('functions', [f'y{i}' for i in range(len(y))])
                
                header = ['t'] + func_names
                rows = []
                for i in range(len(t)):
                    row = [t[i]] + [y[j][i] for j in range(len(y))]
                    rows.append(row)
                
                data = {'numerical_data': {'header': header, 'rows': rows}}
                
            elif 'x' in self.last_solution and 'u' in self.last_solution:
                # PDE
                x = self.last_solution['x']
                t = self.last_solution['t']
                u = self.last_solution['u']
                
                header = ['t', 'x', 'u']
                rows = []
                for i in range(len(t)):
                    for j in range(len(x)):
                        rows.append([t[i], x[j], u[i, j]])
                
                data = {'numerical_data': {'header': header, 'rows': rows}}
            else:
                print("⚠ No numerical data to export")
                return
            
            filename = self.generate_filename("numerical_data", "csv")
            if self.exporter.export_to_csv(data, filename):
                print(f"✅ Exported: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
    
    # ========================================================================
    # HELP &amp; EXAMPLES
    # ========================================================================
    
    def show_help(self):
        """Show help"""
        print("\n" + "=" * 80)
        print("HELP")
        print("=" * 80)
        
        print("\n📖 NOTATION")
        print("-" * 80)
        print("ODEs: y', y'', y'''")
        print("PDEs: u_t, u_x, u_xx, u_tt")
        print("Functions: exp(x), sin(x), cos(x), log(x), sqrt(x), pi, e")
        
        print("\n💾 FILES")
        print("-" * 80)
        print(f"Directory: {os.path.abspath(self.output_dir)}/")
        print("Images: PNG (300 DPI)")
        print("Animations: GIF")
        print("Export: JSON, LaTeX, CSV")
        
        print("\n🎨 FEATURES")
        print("-" * 80)
        print("• Symbolic &amp; numerical solving")
        print("• 2D/3D visualization")
        print("• Animation support")
        print("• Stability analysis")
        print("• Phase portraits")
        print("• Export to multiple formats")
        
        print("\n💡 TIPS")
        print("-" * 80)
        print("• Use explicit multiplication: 2*x")
        print("• Parentheses for clarity: (x+1)/(x-1)")
        print("• ICs determine particular solutions")
        print("• Try 'numerical' for difficult equations")
        print("• Type 'examples' for sample problems")
        
        print("\n" + "=" * 80)
    
    def show_examples(self):
        """Show examples"""
        print("\n" + "=" * 80)
        print("EXAMPLES")
        print("=" * 80)
        
        examples = [
            ("1️⃣  Exponential Growth", "ode", "y' = 2*y", "y(0)=1"),
            ("2️⃣  Harmonic Oscillator", "ode", "y'' + y = 0", "y(0)=1, y'(0)=0"),
            ("3️⃣  Damped Oscillator", "ode", "y'' + 2*y' + 2*y = 0", "y(0)=1, y'(0)=0"),
            ("4️⃣  Circular Motion", "ode_sys", "x' = -y, y' = x", "x(0)=1, y(0)=0"),
            ("5️⃣  Heat Equation", "numerical → 1", "u_t = 0.01*u_xx", ""),
            ("6️⃣  Wave Equation", "numerical → 2", "u_tt = 1.0*u_xx", ""),
            ("7️⃣  Reaction-Diffusion", "numerical → 3", "Pattern formation", ""),
            ("8️⃣  Predator-Prey", "numerical → 4", "Lotka-Volterra", "")
        ]
        
        for title, cmd, eq, ics in examples:
            print(f"\n{title}")
            print("-" * 60)
            print(f"Command: {cmd}")
            print(f"Equation: {eq}")
            if ics:
                print(f"ICs: {ics}")
        
        print("\n" + "=" * 80)
        print(f"💡 All outputs saved to: {self.output_dir}/")
        print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point"""
    try:
        solver = DifferentialEquationSolver()
        solver.run()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        traceback.print_exc()
    finally:
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
