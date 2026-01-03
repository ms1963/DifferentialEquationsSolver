"""
Differential Equation Solver
Supports ODEs, PDEs, and their systems with visualization
Version: 2.0, Michael Stal, 2026
"""

import sympy as sp
from sympy import symbols, Function, Eq, dsolve, sympify
from sympy import exp, sin, cos, tan, log, sqrt, pi, E, I
from sympy import Derivative, Integral
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import re
import warnings
warnings.filterwarnings('ignore')


class DifferentialEquationSolver:
    """Main solver class for differential equations"""
    
    def __init__(self):
        self.last_solution = None
        self.last_type = None
        self.last_equations = None
        
    def run(self):
        """Main application loop"""
        print("=" * 70)
        print(" DIFFERENTIAL EQUATION SOLVER".center(70))
        print("=" * 70)
        print("\nSupports:")
        print("  • Single ODEs and PDEs")
        print("  • Systems of ODEs and PDEs")
        print("  • Visualization and phase portraits")
        print("=" * 70)
        
        while True:
            print("\n" + "=" * 70)
            print("COMMANDS")
            print("=" * 70)
            print("  ode       - Solve a single ODE")
            print("  ode_sys   - Solve a system of ODEs")
            print("  pde       - Solve a single PDE")
            print("  pde_sys   - Solve a system of PDEs")
            print("  visualize - Visualize last solution")
            print("  examples  - Show example problems")
            print("  help      - Show detailed help")
            print("  exit      - Exit program")
            print("=" * 70)
            
            try:
                command = input("\nCommand: ").strip().lower()
                
                if command == 'ode':
                    self.solve_ode()
                elif command == 'ode_sys':
                    self.solve_ode_system()
                elif command == 'pde':
                    self.solve_pde()
                elif command == 'pde_sys':
                    self.solve_pde_system()
                elif command == 'visualize':
                    self.visualize()
                elif command == 'examples':
                    self.show_examples()
                elif command == 'help':
                    self.show_help()
                elif command == 'exit':
                    print("\nThank you for using the Differential Equation Solver!")
                    break
                else:
                    print(f"\n❌ Unknown command: '{command}'")
                    print("   Type 'help' for available commands")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {str(e)}")
                print("   Please try again or type 'help' for assistance")
    
    # ========================================================================
    # SINGLE ODE SOLVER
    # ========================================================================
    
    def solve_ode(self):
        """Solve a single ordinary differential equation"""
        print("\n" + "=" * 70)
        print("SOLVE ODE")
        print("=" * 70)
        
        # Get ODE from user
        print("\nEnter the ODE using notation:")
        print("  y'      for first derivative")
        print("  y''     for second derivative")
        print("  exp(x)  for e^x")
        print("  sin(x), cos(x), log(x), sqrt(x)")
        print("\nExample: y'' + 2*y' + y = exp(x)")
        
        ode_str = input("\nODE: ").strip()
        
        if not ode_str:
            print("❌ No equation entered")
            return
        
        # Get initial conditions
        print("\nInitial conditions (optional, comma-separated):")
        print("  Example: y(0)=1, y'(0)=0")
        ics_str = input("Initial conditions: ").strip()
        
        try:
            # Solve the ODE
            solution = self._solve_ode_internal(ode_str, ics_str)
            
            # Store for visualization
            self.last_solution = solution
            self.last_type = 'ode'
            self.last_equations = ode_str
            
            # Display solution
            print("\n" + "=" * 70)
            print("RESULT")
            print("=" * 70)
            print(f"\n✓ Solution: {solution['solution']}")
            print("\n" + "=" * 70)
            
        except Exception as e:
            print(f"\n❌ Error solving ODE: {str(e)}")
            print("   Please check your equation syntax")
    
    def _solve_ode_internal(self, ode_str, ics_str):
        """Internal method to solve ODE"""
        x = symbols('x')
        y = Function('y')
        
        # Parse the ODE
        ode_eq = self._parse_ode(ode_str, x, y)
        
        print("\n" + "-" * 70)
        print("Solving...")
        print("-" * 70)
        
        # Solve the ODE
        general_solution = dsolve(ode_eq, y(x))
        
        print(f"\nGeneral solution: {general_solution}")
        
        # Apply initial conditions if provided
        if ics_str:
            solution = self._apply_ode_ics(general_solution, ics_str, x, y)
        else:
            solution = general_solution
        
        return {
            'solution': solution,
            'variable': x,
            'function': y,
            'equation': ode_eq
        }
    
    def _parse_ode(self, ode_str, x, y):
        """Parse ODE string into SymPy equation"""
        # Replace derivatives - handle multiple primes
        ode_str = ode_str.replace("y'''", "Derivative(y(x), x, x, x)")
        ode_str = ode_str.replace("y''", "Derivative(y(x), x, x)")
        ode_str = ode_str.replace("y'", "Derivative(y(x), x)")
        
        # Replace y with y(x) - but not in already replaced derivatives
        ode_str = re.sub(r'\by\b(?!\()', 'y(x)', ode_str)
        
        # Split by '='
        if '=' in ode_str:
            left, right = ode_str.split('=', 1)
            left = left.strip()
            right = right.strip()
        else:
            left = ode_str
            right = '0'
        
        # Create namespace
        namespace = {
            'x': x,
            'y': y,
            'exp': exp,
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'log': log,
            'sqrt': sqrt,
            'pi': pi,
            'e': E,
            'E': E,
            'Derivative': Derivative
        }
        
        # Parse both sides
        try:
            left_expr = sympify(left, locals=namespace)
            right_expr = sympify(right, locals=namespace)
        except Exception as e:
            raise ValueError(f"Could not parse equation: {str(e)}")
        
        return Eq(left_expr, right_expr)
    
    def _apply_ode_ics(self, general_solution, ics_str, x, y):
        """Apply initial conditions to ODE solution"""
        print("\n" + "=" * 70)
        print("APPLYING INITIAL CONDITIONS")
        print("=" * 70)
        
        # Extract constants from general solution
        if isinstance(general_solution, Eq):
            rhs = general_solution.rhs
        else:
            rhs = general_solution
        
        constants = sorted(list(rhs.free_symbols - {x}), key=str)
        print(f"Constants in solution: {{{', '.join(map(str, constants))}}} ({len(constants)} total)")
        
        # Parse initial conditions
        ics = []
        ic_parts = [ic.strip() for ic in ics_str.split(',') if ic.strip()]
        print(f"Initial conditions provided: {len(ic_parts)}")
        
        for ic_str in ic_parts:
            ic_str = ic_str.strip()
            
            # Match patterns like y(0)=1 or y'(0)=2 or y''(0)=3
            match = re.match(r"y(\'+)?\(([^)]+)\)\s*=\s*(.+)", ic_str)
            if match:
                deriv_order = len(match.group(1)) if match.group(1) else 0
                point_str = match.group(2)
                value_str = match.group(3)
                
                try:
                    point = float(sympify(point_str))
                    value = float(sympify(value_str))
                    ics.append((deriv_order, point, value))
                    
                    print(f"\n  Processing: {ic_str}")
                    print(f"    Derivative order: {deriv_order}")
                    print(f"    Point: x = {point}")
                    print(f"    Value: {value}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not parse IC '{ic_str}': {str(e)}")
        
        if len(ics) == 0:
            print("\n⚠ No valid initial conditions found")
            return general_solution
        
        if len(ics) != len(constants):
            print(f"\n⚠ Warning: {len(constants)} constants but {len(ics)} initial conditions")
            if len(ics) < len(constants):
                print("  Returning general solution with undetermined constants")
                return general_solution
        
        # Build system of equations
        print("\n" + "=" * 70)
        print("SOLVING FOR CONSTANTS")
        print("=" * 70)
        
        equations = []
        for deriv_order, point, value in ics:
            # Get the appropriate derivative
            expr = rhs
            for _ in range(deriv_order):
                expr = expr.diff(x)
            
            # Substitute the point
            try:
                eq = expr.subs(x, point) - value
                equations.append(eq)
                print(f"    Equation: {expr.subs(x, point)} = {value}")
            except Exception as e:
                print(f"  ⚠ Warning: Could not create equation: {str(e)}")
        
        if len(equations) == 0:
            print("❌ No valid equations created")
            return general_solution
        
        print(f"\nConstants: {constants}")
        print(f"Number of equations: {len(equations)}")
        
        # Solve for constants
        try:
            const_solution = sp.solve(equations, constants)
            print(f"\nRaw solution: {const_solution}")
            
            # Handle different solution formats
            if isinstance(const_solution, dict):
                const_values = const_solution
            elif isinstance(const_solution, list):
                if len(const_solution) > 0:
                    if isinstance(const_solution[0], dict):
                        const_values = const_solution[0]
                    elif isinstance(const_solution[0], tuple):
                        # List of tuples format
                        const_values = dict(zip(constants, const_solution[0]))
                    else:
                        const_values = {}
                else:
                    const_values = {}
            else:
                const_values = {}
            
            if const_values:
                print("\n" + "=" * 70)
                print("RESULT")
                print("=" * 70)
                print(f"✓ Constants determined: {const_values}")
                print("=" * 70)
                
                # Substitute constants back
                particular_solution = rhs.subs(const_values)
                return Eq(y(x), particular_solution)
            else:
                print("\n⚠ Could not determine all constants")
                return general_solution
            
        except Exception as e:
            print(f"\n❌ Could not solve for constants: {str(e)}")
            return general_solution
    
    # ========================================================================
    # SYSTEM OF ODEs SOLVER
    # ========================================================================
    
    def solve_ode_system(self):
        """Solve a system of ordinary differential equations"""
        print("\n" + "=" * 70)
        print("SOLVE SYSTEM OF ODEs")
        print("=" * 70)
        
        # Get number of equations
        n_str = input("\nNumber of equations in the system: ").strip()
        try:
            n = int(n_str)
            if n < 1:
                print("❌ Number must be at least 1")
                return
        except ValueError:
            print(f"❌ Invalid number: {n_str}")
            return
        
        # Get the equations
        equations = []
        functions = []
        print("\nEnter the equations (e.g., x' = 2*x - y, y' = x + 3*y):")
        
        for i in range(n):
            eq_str = input(f"  Equation {i+1}: ").strip()
            if not eq_str:
                print(f"❌ Empty equation {i+1}")
                return
            equations.append(eq_str)
            
            # Extract function name (x, y, z, etc.)
            if "'" in eq_str:
                func_name = eq_str.split("'")[0].strip()
                if func_name not in functions:
                    functions.append(func_name)
        
        if len(functions) == 0:
            print("❌ No functions found in equations")
            return
        
        # Get initial conditions
        print("\nInitial conditions (optional, comma-separated):")
        print("  Example: x(0)=1, y(0)=2")
        ics_str = input("  ICs: ").strip()
        
        try:
            # Solve the system
            solution = self._solve_ode_system_internal(equations, functions, ics_str)
            
            # Store for visualization
            self.last_solution = solution
            self.last_type = 'ode_system'
            self.last_equations = equations
            
            # Display solution
            self._display_ode_system_solution(solution)
            
        except Exception as e:
            print(f"\n❌ Error solving ODE system: {str(e)}")
            print("   Please check your equations")
    
    def _solve_ode_system_internal(self, equations, func_names, ics_str):
        """Internal method to solve ODE system"""
        t = symbols('t')
        
        # Create symbolic functions
        funcs = [Function(f)(t) for f in func_names]
        
        print("\n" + "-" * 70)
        print("Parsing equations...")
        print("-" * 70)
        
        # Parse equations
        eqs = []
        for eq_str in equations:
            try:
                eq = self._parse_system_ode(eq_str, func_names, funcs, t)
                eqs.append(eq)
                print(f"  {eq}")
            except Exception as e:
                print(f"❌ Error parsing equation '{eq_str}': {str(e)}")
                raise
        
        print("\n" + "-" * 70)
        print("Solving system...")
        print("-" * 70)
        
        # Solve the system
        try:
            solution = dsolve(eqs, funcs)
            
            # Ensure solution is a list
            if not isinstance(solution, list):
                solution = [solution]
            
            print(f"\nGeneral solution found with {len(solution)} equation(s)")
            
            # Apply initial conditions if provided
            if ics_str:
                solution = self._apply_system_ics(solution, ics_str, func_names, funcs, t)
            
            return {
                'solution': solution,
                'functions': func_names,
                'variable': t,
                'equations': eqs
            }
            
        except Exception as e:
            print(f"❌ Could not solve system analytically: {str(e)}")
            raise
    
    def _parse_system_ode(self, eq_str, func_names, funcs, var):
        """Parse a single equation in ODE system"""
        # Split by '='
        if '=' not in eq_str:
            raise ValueError(f"Equation must contain '=': {eq_str}")
        
        left, right = eq_str.split('=', 1)
        left = left.strip()
        right = right.strip()
        
        # Handle left side (derivative)
        if "'" in left:
            func_name = left.replace("'", "").strip()
            if func_name not in func_names:
                raise ValueError(f"Unknown function: {func_name}")
            idx = func_names.index(func_name)
            left_expr = funcs[idx].diff(var)
        else:
            # Just the function itself
            if left not in func_names:
                raise ValueError(f"Unknown function: {left}")
            idx = func_names.index(left)
            left_expr = funcs[idx]
        
        # Parse right side
        right_expr = self._parse_system_expression(right, func_names, funcs, var)
        
        return Eq(left_expr, right_expr)
    
    def _parse_system_expression(self, expr_str, func_names, funcs, var):
        """Parse expression with multiple functions"""
        # Replace function names with symbolic functions
        parsed = expr_str
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_names = sorted(func_names, key=len, reverse=True)
        
        for name in sorted_names:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(name) + r'\b'
            replacement = f'{name}({var})'
            parsed = re.sub(pattern, replacement, parsed)
        
        # Create namespace
        namespace = {
            str(var): var,
            'exp': exp,
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'log': log,
            'sqrt': sqrt,
            'pi': pi,
            'e': E,
            'E': E
        }
        
        # Add functions to namespace
        for i, name in enumerate(func_names):
            namespace[f'{name}({var})'] = funcs[i]
        
        try:
            return sympify(parsed, locals=namespace)
        except Exception as e:
            raise ValueError(f"Could not parse expression '{expr_str}': {str(e)}")
    
    def _apply_system_ics(self, solution, ics_str, func_names, funcs, var):
        """Apply initial conditions to system solution"""
        print("\n" + "=" * 70)
        print("APPLYING INITIAL CONDITIONS")
        print("=" * 70)
        
        # Extract all constants
        constants = set()
        for sol in solution:
            constants.update(sol.rhs.free_symbols - {var})
        
        constants = sorted(list(constants), key=str)
        print(f"Constants: {constants} ({len(constants)} total)")
        
        # Parse initial conditions
        ics = {}
        ic_parts = [ic.strip() for ic in ics_str.split(',') if ic.strip()]
        
        for ic_str in ic_parts:
            # Match pattern: x(0)=1
            match = re.match(r'(\w+)\(([^)]+)\)\s*=\s*(.+)', ic_str)
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
                        print(f"  {func_name}({point}) = {value}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not parse IC '{ic_str}': {str(e)}")
        
        # Build equations
        equations = []
        for func, (point, value) in ics.items():
            # Find corresponding solution
            for sol in solution:
                if sol.lhs == func:
                    try:
                        eq = sol.rhs.subs(var, point) - value
                        equations.append(eq)
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not create equation: {str(e)}")
        
        if len(equations) == 0:
            print("⚠ No valid initial conditions found")
            return solution
        
        print(f"\nSolving {len(equations)} equation(s) for {len(constants)} constant(s)...")
        
        # Solve for constants
        try:
            const_values = sp.solve(equations, constants)
            
            # Handle different solution formats
            if isinstance(const_values, list) and len(const_values) > 0:
                if isinstance(const_values[0], dict):
                    const_values = const_values[0]
                elif isinstance(const_values[0], tuple):
                    const_values = dict(zip(constants, const_values[0]))
            
            if const_values:
                print(f"✓ Constants: {const_values}")
                
                # Substitute back
                final_solution = []
                for sol in solution:
                    new_rhs = sol.rhs.subs(const_values)
                    final_solution.append(Eq(sol.lhs, new_rhs))
                
                return final_solution
            else:
                print("⚠ Could not determine all constants")
                return solution
            
        except Exception as e:
            print(f"❌ Could not determine constants: {str(e)}")
            return solution
    
    def _display_ode_system_solution(self, solution_dict):
        """Display the solution of ODE system"""
        print("\n" + "=" * 70)
        print("SYSTEM SOLUTION")
        print("=" * 70)
        
        solution = solution_dict['solution']
        func_names = solution_dict['functions']
        
        for i, sol in enumerate(solution):
            if i < len(func_names):
                print(f"\n✓ {func_names[i]}(t) = {sol.rhs}")
            else:
                print(f"\n✓ {sol}")
        
        print("\n" + "=" * 70)
    
    # ========================================================================
    # SINGLE PDE SOLVER
    # ========================================================================
    
    def solve_pde(self):
        """Solve a single partial differential equation"""
        print("\n" + "=" * 70)
        print("SOLVE PDE")
        print("=" * 70)
        
        print("\nEnter the PDE using notation:")
        print("  u_t   for ∂u/∂t")
        print("  u_x   for ∂u/∂x")
        print("  u_xx  for ∂²u/∂x²")
        print("\nExample: u_t = alpha*u_xx  (Heat equation)")
        
        pde_str = input("\nPDE: ").strip()
        
        if not pde_str:
            print("❌ No equation entered")
            return
        
        print("\nBoundary/Initial conditions (optional):")
        print("  Example: u(x,0)=sin(pi*x), u(0,t)=0, u(1,t)=0")
        conditions_str = input("Conditions: ").strip()
        
        try:
            solution = self._solve_pde_internal(pde_str, conditions_str)
            
            self.last_solution = solution
            self.last_type = 'pde'
            self.last_equations = pde_str
            
            print("\n" + "=" * 70)
            print("RESULT")
            print("=" * 70)
            print(f"\n✓ Solution: {solution['solution']}")
            print("\n" + "=" * 70)
            
        except Exception as e:
            print(f"\n❌ Error solving PDE: {str(e)}")
            print("   Please check your equation syntax")
    
    def _solve_pde_internal(self, pde_str, conditions_str):
        """Internal method to solve PDE"""
        x, t = symbols('x t')
        u = Function('u')
        
        # Parse the PDE
        pde_eq = self._parse_pde(pde_str, x, t, u)
        
        print("\n" + "-" * 70)
        print(f"Equation: {pde_eq}")
        print("Solving...")
        print("-" * 70)
        
        # Try to solve
        try:
            from sympy import pdsolve
            solution = pdsolve(pde_eq, u(x, t))
            
            print(f"\nGeneral solution: {solution}")
            
            return {
                'solution': solution,
                'variables': (x, t),
                'function': u,
                'equation': pde_eq
            }
            
        except Exception as e:
            print(f"⚠ Analytical solution not available: {str(e)}")
            print("  Returning symbolic form")
            
            return {
                'solution': pde_eq,
                'variables': (x, t),
                'function': u,
                'equation': pde_eq
            }
    
    def _parse_pde(self, pde_str, x, t, u):
        """Parse PDE string into SymPy equation"""
        # Replace partial derivatives
        pde_str = pde_str.replace("u_tt", "Derivative(u(x,t), t, t)")
        pde_str = pde_str.replace("u_xx", "Derivative(u(x,t), x, x)")
        pde_str = pde_str.replace("u_xt", "Derivative(u(x,t), x, t)")
        pde_str = pde_str.replace("u_tx", "Derivative(u(x,t), t, x)")
        pde_str = pde_str.replace("u_t", "Derivative(u(x,t), t)")
        pde_str = pde_str.replace("u_x", "Derivative(u(x,t), x)")
        pde_str = re.sub(r'\bu\b(?!\()', 'u(x,t)', pde_str)
        
        # Split by '='
        if '=' in pde_str:
            left, right = pde_str.split('=', 1)
        else:
            left = pde_str
            right = '0'
        
        # Create namespace
        namespace = {
            'x': x,
            't': t,
            'u': u,
            'exp': exp,
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'log': log,
            'sqrt': sqrt,
            'pi': pi,
            'e': E,
            'E': E,
            'Derivative': Derivative,
            'alpha': symbols('alpha'),
            'beta': symbols('beta'),
            'gamma': symbols('gamma'),
            'c': symbols('c')
        }
        
        try:
            left_expr = sympify(left.strip(), locals=namespace)
            right_expr = sympify(right.strip(), locals=namespace)
        except Exception as e:
            raise ValueError(f"Could not parse PDE: {str(e)}")
        
        return Eq(left_expr, right_expr)
    
    # ========================================================================
    # SYSTEM OF PDEs SOLVER
    # ========================================================================
    
    def solve_pde_system(self):
        """Solve a system of partial differential equations"""
        print("\n" + "=" * 70)
        print("SOLVE SYSTEM OF PDEs")
        print("=" * 70)
        
        n_str = input("\nNumber of equations in the system: ").strip()
        try:
            n = int(n_str)
            if n < 1:
                print("❌ Number must be at least 1")
                return
        except ValueError:
            print(f"❌ Invalid number: {n_str}")
            return
        
        equations = []
        functions = []
        
        print("\nEnter the equations using notation:")
        print("  u_t for ∂u/∂t, u_xx for ∂²u/∂x²")
        print("  Example: u_t = alpha*u_xx + beta*v")
        
        for i in range(n):
            eq_str = input(f"  Equation {i+1}: ").strip()
            if not eq_str:
                print(f"❌ Empty equation {i+1}")
                return
            equations.append(eq_str)
            
            # Extract function name
            if "_" in eq_str:
                func_name = eq_str.split("_")[0].strip()
                if func_name not in functions:
                    functions.append(func_name)
        
        if len(functions) == 0:
            print("❌ No functions found in equations")
            return
        
        print("\nBoundary/Initial conditions (optional):")
        conditions_str = input("Conditions: ").strip()
        
        try:
            solution = self._solve_pde_system_internal(equations, functions, conditions_str)
            
            self.last_solution = solution
            self.last_type = 'pde_system'
            self.last_equations = equations
            
            self._display_pde_system_solution(solution)
            
        except Exception as e:
            print(f"\n❌ Error solving PDE system: {str(e)}")
            print("   Please check your equations")
    
    def _solve_pde_system_internal(self, equations, func_names, conditions_str):
        """Internal method to solve PDE system"""
        x, t = symbols('x t')
        
        # Create symbolic functions
        funcs = [Function(f)(x, t) for f in func_names]
        
        print("\n" + "-" * 70)
        print("Parsing equations...")
        print("-" * 70)
        
        # Parse equations
        eqs = []
        for eq_str in equations:
            try:
                eq = self._parse_system_pde(eq_str, func_names, funcs, x, t)
                eqs.append(eq)
                print(f"  {eq}")
            except Exception as e:
                print(f"❌ Error parsing equation '{eq_str}': {str(e)}")
                raise
        
        print("\n⚠ Note: Analytical solutions for PDE systems are often not available")
        print("  Returning symbolic form")
        
        return {
            'equations': eqs,
            'functions': func_names,
            'variables': (x, t),
            'symbolic_funcs': funcs
        }
    
    def _parse_system_pde(self, eq_str, func_names, funcs, x, t):
        """Parse a single equation in PDE system"""
        # Process each function's derivatives
        for i, name in enumerate(func_names):
            eq_str = eq_str.replace(f"{name}_tt", f"Derivative({name}(x,t), t, t)")
            eq_str = eq_str.replace(f"{name}_xx", f"Derivative({name}(x,t), x, x)")
            eq_str = eq_str.replace(f"{name}_xt", f"Derivative({name}(x,t), x, t)")
            eq_str = eq_str.replace(f"{name}_t", f"Derivative({name}(x,t), t)")
            eq_str = eq_str.replace(f"{name}_x", f"Derivative({name}(x,t), x)")
            pattern = r'\b' + re.escape(name) + r'\b(?!\()'
            eq_str = re.sub(pattern, f'{name}(x,t)', eq_str)
        
        # Split by '='
        if '=' not in eq_str:
            raise ValueError(f"Equation must contain '=': {eq_str}")
        
        left, right = eq_str.split('=', 1)
        
        # Create namespace
        namespace = {
            'x': x,
            't': t,
            'exp': exp,
            'sin': sin,
            'cos': cos,
            'Derivative': Derivative,
            'alpha': symbols('alpha'),
            'beta': symbols('beta'),
            'gamma': symbols('gamma'),
            'delta': symbols('delta'),
            'D1': symbols('D1'),
            'D2': symbols('D2'),
            'c': symbols('c')
        }
        
        # Add functions
        for i, name in enumerate(func_names):
            namespace[f'{name}(x,t)'] = funcs[i]
        
        try:
            left_expr = sympify(left.strip(), locals=namespace)
            right_expr = sympify(right.strip(), locals=namespace)
        except Exception as e:
            raise ValueError(f"Could not parse equation: {str(e)}")
        
        return Eq(left_expr, right_expr)
    
    def _display_pde_system_solution(self, solution_dict):
        """Display PDE system (usually symbolic)"""
        print("\n" + "=" * 70)
        print("SYSTEM OF PDEs")
        print("=" * 70)
        
        equations = solution_dict['equations']
        func_names = solution_dict['functions']
        
        print("\nSymbolic form:")
        for i, eq in enumerate(equations):
            if i < len(func_names):
                print(f"\n  {func_names[i]}: {eq}")
            else:
                print(f"\n  {eq}")
        
        print("\n" + "=" * 70)
        print("Note: Analytical solutions for PDE systems often require")
        print("      numerical methods (finite differences, spectral methods, etc.)")
        print("=" * 70)
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    def visualize(self):
        """Visualize the last solution"""
        if self.last_solution is None:
            print("\n❌ No solution to visualize. Solve an equation first.")
            return
        
        print("\n" + "=" * 70)
        print("VISUALIZING SOLUTION")
        print("=" * 70)
        
        try:
            if self.last_type == 'ode':
                self._visualize_ode(self.last_solution)
            elif self.last_type == 'ode_system':
                self._visualize_ode_system(self.last_solution)
            elif self.last_type == 'pde':
                self._visualize_pde(self.last_solution)
            elif self.last_type == 'pde_system':
                print("\n⚠ Visualization of PDE systems not yet implemented")
                print("  Please use numerical methods for visualization")
            else:
                print(f"\n❌ Unknown solution type: {self.last_type}")
                
        except Exception as e:
            print(f"\n❌ Error during visualization: {str(e)}")
    
    def _visualize_ode(self, solution_dict):
        """Visualize single ODE solution"""
        from sympy import lambdify
        
        solution = solution_dict['solution']
        var = solution_dict['variable']
        
        # Extract the solution expression
        if isinstance(solution, Eq):
            expr = solution.rhs
        else:
            expr = solution
        
        # Check if solution contains undetermined constants
        free_symbols = expr.free_symbols - {var}
        if free_symbols:
            print(f"\n⚠ Solution contains undetermined constants: {free_symbols}")
            print("  Cannot visualize without specific values.")
            print("  Please provide initial conditions.")
            return
        
        # Create numerical function
        try:
            func = lambdify(var, expr, modules=['numpy'])
        except Exception as e:
            print(f"❌ Could not create numerical function: {str(e)}")
            return
        
        # Generate plot
        x_vals = np.linspace(0, 10, 1000)
        
        try:
            y_vals = func(x_vals)
            
            # Check for invalid values
            if np.any(np.isnan(y_vals)) or np.any(np.isinf(y_vals)):
                print("⚠ Warning: Solution contains NaN or Inf values")
                # Filter out invalid values
                valid_mask = np.isfinite(y_vals)
                x_vals = x_vals[valid_mask]
                y_vals = y_vals[valid_mask]
                
                if len(x_vals) == 0:
                    print("❌ No valid points to plot")
                    return
                    
        except Exception as e:
            print(f"❌ Error evaluating function: {str(e)}")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        
        # Plot solution
        axes[0].plot(x_vals, y_vals, 'b-', linewidth=2)
        axes[0].set_xlabel('x', fontsize=12)
        axes[0].set_ylabel('y(x)', fontsize=12)
        axes[0].set_title('Solution y(x)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot first derivative
        try:
            expr_prime = expr.diff(var)
            func_prime = lambdify(var, expr_prime, modules=['numpy'])
            y_prime_vals = func_prime(x_vals)
            
            axes[1].plot(x_vals, y_prime_vals, 'r-', linewidth=2)
            axes[1].set_xlabel('x', fontsize=12)
            axes[1].set_ylabel("y'(x)", fontsize=12)
            axes[1].set_title("First Derivative y'(x)", fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
        except Exception as e:
            print(f"⚠ Could not plot first derivative: {str(e)}")
            axes[1].text(0.5, 0.5, 'Could not compute derivative', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        # Plot second derivative
        try:
            expr_double_prime = expr_prime.diff(var)
            func_double_prime = lambdify(var, expr_double_prime, modules=['numpy'])
            y_double_prime_vals = func_double_prime(x_vals)
            
            axes[2].plot(x_vals, y_double_prime_vals, 'g-', linewidth=2)
            axes[2].set_xlabel('x', fontsize=12)
            axes[2].set_ylabel("y''(x)", fontsize=12)
            axes[2].set_title("Second Derivative y''(x)", fontsize=14, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
        except Exception as e:
            print(f"⚠ Could not plot second derivative: {str(e)}")
            axes[2].text(0.5, 0.5, 'Could not compute derivative', 
                        ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Visualization complete")
    
    def _visualize_ode_system(self, solution_dict):
        """Visualize ODE system solution"""
        from sympy import lambdify
        
        solution = solution_dict['solution']
        func_names = solution_dict['functions']
        var = solution_dict['variable']
        
        # Check for undetermined constants
        for sol in solution:
            free_symbols = sol.rhs.free_symbols - {var}
            if free_symbols:
                print(f"\n⚠ Solution contains undetermined constants: {free_symbols}")
                print("  Please provide initial conditions.")
                return
        
        # Create numerical functions
        funcs = []
        for sol in solution:
            try:
                f = lambdify(var, sol.rhs, modules=['numpy'])
                funcs.append(f)
            except Exception as e:
                print(f"❌ Error creating function: {str(e)}")
                return
        
        # Generate values
        t_vals = np.linspace(0, 10, 1000)
        
        # Determine number of plots
        n_funcs = len(funcs)
        n_plots = n_funcs + (1 if n_funcs == 2 else 0)  # Add phase portrait for 2D
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
        
        if n_plots == 1:
            axes = [axes]
        
        # Plot each function
        all_vals = []
        for i, (func, name) in enumerate(zip(funcs, func_names)):
            try:
                vals = func(t_vals)
                
                # Check for invalid values
                if np.any(np.isnan(vals)) or np.any(np.isinf(vals)):
                    print(f"⚠ Warning: {name}(t) contains NaN or Inf values")
                    valid_mask = np.isfinite(vals)
                    vals = np.where(valid_mask, vals, 0)
                
                all_vals.append(vals)
                
                axes[i].plot(t_vals, vals, linewidth=2)
                axes[i].set_xlabel('t', fontsize=12)
                axes[i].set_ylabel(f'{name}(t)', fontsize=12)
                axes[i].set_title(f'Solution: {name}(t)', fontsize=14, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
            except Exception as e:
                print(f"❌ Error plotting {name}: {str(e)}")
                axes[i].text(0.5, 0.5, f'Error plotting {name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        # Add phase portrait for 2D systems
        if n_funcs == 2 and len(all_vals) == 2:
            try:
                axes[-1].plot(all_vals[0], all_vals[1], 'purple', linewidth=2)
                axes[-1].plot(all_vals[0][0], all_vals[1][0], 'go', markersize=10, label='Start')
                axes[-1].plot(all_vals[0][-1], all_vals[1][-1], 'ro', markersize=10, label='End')
                axes[-1].set_xlabel(f'{func_names[0]}(t)', fontsize=12)
                axes[-1].set_ylabel(f'{func_names[1]}(t)', fontsize=12)
                axes[-1].set_title('Phase Portrait', fontsize=14, fontweight='bold')
                axes[-1].grid(True, alpha=0.3)
                axes[-1].legend()
            except Exception as e:
                print(f"⚠ Could not create phase portrait: {str(e)}")
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Visualization complete")
    
    def _visualize_pde(self, solution_dict):
        """Visualize PDE solution (if possible)"""
        print("\n⚠ PDE visualization requires specific boundary/initial conditions")
        print("  and is problem-dependent. Showing symbolic solution instead.")
        
        solution = solution_dict['solution']
        print(f"\nSolution: {solution}")
    
    # ========================================================================
    # HELP AND EXAMPLES
    # ========================================================================
    
    def show_help(self):
        """Show detailed help information"""
        print("\n" + "=" * 70)
        print("HELP - DIFFERENTIAL EQUATION SOLVER")
        print("=" * 70)
        
        print("\n📖 NOTATION GUIDE")
        print("-" * 70)
        print("\nFor ODEs:")
        print("  y'      First derivative dy/dx")
        print("  y''     Second derivative d²y/dx²")
        print("  y'''    Third derivative d³y/dx³")
        print("\nFor PDEs:")
        print("  u_t     Partial derivative ∂u/∂t")
        print("  u_x     Partial derivative ∂u/∂x")
        print("  u_xx    Second partial ∂²u/∂x²")
        print("  u_tt    Second partial ∂²u/∂t²")
        print("\nFunctions:")
        print("  exp(x)  Exponential e^x")
        print("  sin(x)  Sine function")
        print("  cos(x)  Cosine function")
        print("  log(x)  Natural logarithm")
        print("  sqrt(x) Square root")
        print("  pi      Pi constant (3.14159...)")
        print("  e or E  Euler's number (2.71828...)")
        
        print("\n📝 EXAMPLES")
        print("-" * 70)
        print("\nSingle ODE:")
        print("  y'' + 2*y' + y = exp(x)")
        print("  Initial conditions: y(0)=1, y'(0)=0")
        
        print("\nSystem of ODEs:")
        print("  x' = -y")
        print("  y' = x")
        print("  Initial conditions: x(0)=1, y(0)=0")
        
        print("\nSingle PDE:")
        print("  u_t = alpha*u_xx  (Heat equation)")
        
        print("\nSystem of PDEs:")
        print("  u_t = D1*u_xx + beta*v")
        print("  v_t = D2*v_xx + gamma*u")
        
        print("\n💡 TIPS")
        print("-" * 70)
        print("  • Use parentheses to clarify order of operations")
        print("  • Multiplication must be explicit: 2*x not 2x")
        print("  • Initial conditions help determine specific solutions")
        print("  • Type 'examples' to see more detailed examples")
        
        print("\n" + "=" * 70)
    
    def show_examples(self):
        """Show example problems"""
        print("\n" + "=" * 70)
        print("EXAMPLE PROBLEMS")
        print("=" * 70)
        
        print("\n1️⃣  SIMPLE HARMONIC OSCILLATOR")
        print("-" * 70)
        print("Command: ode")
        print("ODE: y'' + y = 0")
        print("ICs: y(0)=1, y'(0)=0")
        print("Solution: y(x) = cos(x)")
        print("\nDescription: Undamped oscillation")
        
        print("\n2️⃣  EXPONENTIAL GROWTH")
        print("-" * 70)
        print("Command: ode")
        print("ODE: y' = 2*y")
        print("ICs: y(0)=1")
        print("Solution: y(x) = exp(2*x)")
        print("\nDescription: Population growth model")
        
        print("\n3️⃣  DAMPED OSCILLATOR")
        print("-" * 70)
        print("Command: ode")
        print("ODE: y'' + 2*y' + 2*y = exp(-x)*sin(x)")
        print("ICs: y(0)=0, y'(0)=1")
        print("\nDescription: Forced damped oscillation")
        
        print("\n4️⃣  CIRCULAR MOTION (System)")
        print("-" * 70)
        print("Command: ode_sys")
        print("Number of equations: 2")
        print("Equation 1: x' = -y")
        print("Equation 2: y' = x")
        print("ICs: x(0)=1, y(0)=0")
        print("Solution: x(t) = cos(t), y(t) = sin(t)")
        print("\nDescription: Uniform circular motion")
        
        print("\n5️⃣  PREDATOR-PREY (Lotka-Volterra)")
        print("-" * 70)
        print("Command: ode_sys")
        print("Number of equations: 2")
        print("Equation 1: x' = x - x*y")
        print("Equation 2: y' = -y + x*y")
        print("ICs: x(0)=2, y(0)=1")
        print("\nDescription: Population dynamics (x=prey, y=predator)")
        
        print("\n6️⃣  HEAT EQUATION (PDE)")
        print("-" * 70)
        print("Command: pde")
        print("PDE: u_t = alpha*u_xx")
        print("\nDescription: Heat diffusion in 1D")
        
        print("\n7️⃣  WAVE EQUATION (PDE)")
        print("-" * 70)
        print("Command: pde")
        print("PDE: u_tt = c**2*u_xx")
        print("\nDescription: Wave propagation in 1D")
        
        print("\n8️⃣  FIRST ORDER LINEAR")
        print("-" * 70)
        print("Command: ode")
        print("ODE: y' + y = x")
        print("ICs: y(0)=1")
        print("\nDescription: First-order linear ODE")
        
        print("\n9️⃣  SECOND ORDER CONSTANT COEFFICIENTS")
        print("-" * 70)
        print("Command: ode")
        print("ODE: y'' - 3*y' + 2*y = 0")
        print("ICs: y(0)=1, y'(0)=0")
        print("\nDescription: Homogeneous second-order ODE")
        
        print("\n🔟  COUPLED HARMONIC OSCILLATORS")
        print("-" * 70)
        print("Command: ode_sys")
        print("Number of equations: 2")
        print("Equation 1: x' = y")
        print("Equation 2: y' = -x")
        print("ICs: x(0)=1, y(0)=0")
        print("\nDescription: Two coupled oscillators")
        
        print("\n" + "=" * 70)
        print("TIP: Try these examples to learn how to use the solver!")
        print("=" * 70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        solver = DifferentialEquationSolver()
        solver.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
