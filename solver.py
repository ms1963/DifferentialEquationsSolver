#!/usr/bin/env python3
"""
Differential Equation Solver - Production Grade
Version 5.2 - Fixed Visualization with Order Terms
Supports ODEs and PDEs with advanced visualization
Author:  Michael Stal
License: MIT
"""

import sympy as sp
from sympy import (symbols, Function, Eq, dsolve, pde_separate, pdsolve,
                   sin, cos, tan, exp, log, sqrt, I, pi, E, oo,
                   sinh, cosh, tanh, latex)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor)
from sympy.series.order import Order
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter
import re
import sys
import os
import platform
import threading
import signal
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib backend
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")


def timeout(seconds=300):
    """Decorator to add timeout to functions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if platform.system() != 'Windows':
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(seconds)
                except (ValueError, OSError):
                    pass
            try:
                result = func(*args, **kwargs)
            finally:
                if platform.system() != 'Windows':
                    try:
                        signal.alarm(0)
                    except (ValueError, OSError):
                        pass
            return result
        return wrapper
    return decorator


class ValidationError(Exception):
    """Custom validation exception"""
    pass


class ParseError(Exception):
    """Custom parsing exception"""
    pass


class SolveError(Exception):
    """Custom solving exception"""
    pass


def sanitize_input(input_str: str, max_length: int = 10000) -> str:
    """
    Sanitize user input to prevent code injection
    
    Args:
        input_str: User input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValidationError: If input is invalid
    """
    if not isinstance(input_str, str):
        raise ValidationError("Input must be a string")
    
    if len(input_str) > max_length:
        raise ValidationError(f"Input exceeds maximum length of {max_length}")
    
    # Allow only safe characters
    allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                       '0123456789+-*/()[]{}=.,_\' \t\n*')
    
    for char in input_str:
        if char not in allowed_chars:
            raise ValidationError(f"Input contains invalid character: {char}")
    
    # Check for dangerous patterns
    dangerous = ['__', 'import', 'eval', 'exec', 'compile', 'open', 'file']
    lower_input = input_str.lower()
    for pattern in dangerous:
        if pattern in lower_input:
            raise ValidationError(f"Input contains forbidden pattern: {pattern}")
    
    return input_str.strip()


def validate_number(value: Any, min_val: float = -1e10, max_val: float = 1e10,
                   allow_none: bool = False) -> Optional[float]:
    """
    Validate numeric input
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        allow_none: Whether None is allowed
        
    Returns:
        Validated float value
        
    Raises:
        ValidationError: If value is invalid
    """
    if value is None and allow_none:
        return None
    
    try:
        num = float(value)
    except (ValueError, TypeError):
        raise ValidationError(f"Invalid number: {value}")
    
    if np.isnan(num):
        raise ValidationError("Value is NaN")
    
    if np.isinf(num):
        raise ValidationError("Value is infinite")
    
    if num < min_val or num > max_val:
        raise ValidationError(f"Value {num} out of range [{min_val}, {max_val}]")
    
    return num


def validate_identifier(name: str, max_length: int = 50) -> str:
    """
    Validate identifier (variable/function name)
    
    Args:
        name: Identifier to validate
        max_length: Maximum length
        
    Returns:
        Validated identifier
        
    Raises:
        ValidationError: If identifier is invalid
    """
    if not isinstance(name, str):
        raise ValidationError("Identifier must be a string")
    
    name = name.strip()
    
    if not name:
        raise ValidationError("Identifier cannot be empty")
    
    if len(name) > max_length:
        raise ValidationError(f"Identifier too long (max {max_length})")
    
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
        raise ValidationError(f"Invalid identifier: {name}")
    
    return name


class DifferentialEquationSolver:
    """Production-grade differential equation solver with visualization"""
    
    def __init__(self):
        """Initialize the solver"""
        # Define symbolic variables
        self.x = symbols('x', real=True)
        self.y_var = symbols('y', real=True)
        self.z = symbols('z', real=True)
        self.t = symbols('t', real=True)
        
        # Constants
        self.C1, self.C2, self.C3, self.C4 = symbols('C1 C2 C3 C4')
        
        # Common parameters
        self.alpha = symbols('alpha', positive=True, real=True)
        self.beta = symbols('beta', real=True)
        self.c = symbols('c', positive=True, real=True)
        self.k = symbols('k', real=True)
        self.omega = symbols('omega', real=True)
        
        # Parser transformations
        self.transformations = (
            standard_transformations + 
            (implicit_multiplication_application, convert_xor)
        )
        
        # Thread lock for matplotlib
        self._plot_lock = threading.Lock()
        
        # Backend info
        self.backend = matplotlib.get_backend()
        self.headless = self.backend == 'Agg'
    
    @timeout(300)
    def solve_ode(self, equation_str: str, function_name: str = 'y',
                  independent_var: str = 'x',
                  initial_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve an ordinary differential equation
        
        Args:
            equation_str: The ODE as a string
            function_name: Name of the dependent variable
            independent_var: Name of the independent variable
            initial_conditions: Dictionary of initial conditions
            
        Returns:
            Dictionary containing solution and metadata
        """
        try:
            # Sanitize inputs
            equation_str = sanitize_input(equation_str)
            function_name = validate_identifier(function_name)
            independent_var = validate_identifier(independent_var)
            
            # Define symbols
            x = symbols(independent_var, real=True)
            y = Function(function_name)
            
            # Parse the equation
            equation = self._parse_ode_equation(
                equation_str, y, x, function_name, independent_var
            )
            
            # Solve the ODE
            try:
                solution = dsolve(equation, y(x), simplify=True)
            except NotImplementedError:
                raise SolveError("This ODE type is not supported")
            except Exception as e:
                raise SolveError(f"Failed to solve ODE: {str(e)}")
            
            if solution is None:
                raise SolveError("No solution found")
            
            # Handle multiple solutions
            if isinstance(solution, list):
                if len(solution) == 0:
                    raise SolveError("No solution found")
                solution = solution[0]
            
            # Classify the ODE
            ode_type = self._classify_ode(equation, y(x))
            
            # Create result dict
            result = {
                'success': True,
                'equation': equation,
                'solution': solution,
                'type': ode_type,
                'function': y,
                'variable': x,
                'function_name': function_name,
                'variable_name': independent_var
            }
            
            # Apply initial conditions if provided
            if initial_conditions:
                solution = self._apply_initial_conditions_fixed(
                    solution, initial_conditions, y, x, function_name, independent_var
                )
                result['solution'] = solution
            else:
                # No initial conditions provided - suggest what's needed
                self.suggest_initial_conditions(result)
            
            return result
            
        except (ValidationError, ParseError, SolveError) as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
        except TimeoutError:
            return {
                'success': False,
                'error': 'Operation timed out (>5 minutes)',
                'error_type': 'TimeoutError'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}',
                'error_type': type(e).__name__
            }
    
    def _apply_initial_conditions_fixed(self, solution, ics: Dict, y, x, 
                                       func_name: str, var_name: str):
        """
        Apply initial conditions to solve for constants
        
        Args:
            solution: SymPy solution equation
            ics: Dictionary of initial conditions
            y: Function symbol
            x: Variable symbol
            func_name: Function name string
            var_name: Variable name string
            
        Returns:
            Solution with constants determined
        """
        if not isinstance(solution, Eq):
            return solution
        
        # Get the right-hand side of the solution
        sol_expr = solution.rhs
        
        # Find all constants in the solution
        constants = sol_expr.free_symbols - {x}
        
        if not constants:
            print("\n✓ Solution has no arbitrary constants")
            return solution
        
        if not ics:
            print(f"\n⚠ Solution contains {len(constants)} constant(s): {constants}")
            print(f"  Provide {len(constants)} initial condition(s) to determine them uniquely")
            return solution
        
        num_constants = len(constants)
        num_conditions = len(ics)
        
        print(f"\n{'='*70}")
        print(f"APPLYING INITIAL CONDITIONS")
        print(f"{'='*70}")
        print(f"Constants in solution: {constants} ({num_constants} total)")
        print(f"Initial conditions provided: {num_conditions}")
        
        if num_conditions < num_constants:
            print(f"\n⚠ WARNING: Insufficient initial conditions!")
            print(f"  Need {num_constants} conditions, but only {num_conditions} provided")
            print(f"  Solution will contain {num_constants - num_conditions} arbitrary constant(s)")
        elif num_conditions > num_constants:
            print(f"\n⚠ WARNING: Over-determined system!")
            print(f"  Need {num_constants} conditions, but {num_conditions} provided")
            print(f"  Using first {num_constants} conditions")
        
        # Build system of equations from initial conditions
        equations = []
        
        for ic_key, ic_value in ics.items():
            try:
                # Parse the initial condition
                derivative_order = ic_key.count("'")
                
                # Extract the point value
                x_val = self._extract_ic_point(ic_key, var_name)
                
                print(f"\n  Processing: {ic_key} = {ic_value}")
                print(f"    Derivative order: {derivative_order}")
                print(f"    Point: {var_name} = {x_val}")
                
                # Get the appropriate derivative
                if derivative_order == 0:
                    # Function value: y(x0) = value
                    expr_at_point = sol_expr.subs(x, x_val)
                else:
                    # Derivative: y'(x0) = value, y''(x0) = value, etc.
                    expr_deriv = sol_expr
                    for _ in range(derivative_order):
                        expr_deriv = sp.diff(expr_deriv, x)
                    expr_at_point = expr_deriv.subs(x, x_val)
                
                # Create equation: expr_at_point = ic_value
                eq = Eq(expr_at_point, ic_value)
                equations.append(eq)
                print(f"    Equation: {eq}")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                continue
        
        if not equations:
            print("\n✗ No valid equations created from initial conditions")
            return solution
        
        if len(equations) < num_constants:
            print(f"\n⚠ Only {len(equations)} valid equation(s) for {num_constants} constant(s)")
        
        # Solve the system for constants
        try:
            const_list = list(constants)
            
            print(f"\n{'='*70}")
            print(f"SOLVING FOR CONSTANTS")
            print(f"{'='*70}")
            print(f"Constants: {const_list}")
            print(f"Equations: {equations}")
            
            # Solve the system
            const_solution = sp.solve(equations, const_list, dict=True)
            
            print(f"\nRaw solution: {const_solution}")
            
            if const_solution:
                # Get the first solution
                if isinstance(const_solution, list) and len(const_solution) > 0:
                    const_values = const_solution[0]
                elif isinstance(const_solution, dict):
                    const_values = const_solution
                else:
                    print("✗ Unexpected solution format")
                    return solution
                
                # Check if all constants were solved
                solved_constants = set(const_values.keys())
                unsolved_constants = constants - solved_constants
                
                print(f"\n{'='*70}")
                print(f"RESULT")
                print(f"{'='*70}")
                
                if unsolved_constants:
                    print(f"⚠ Partially solved:")
                    print(f"  Determined: {solved_constants}")
                    print(f"  Undetermined: {unsolved_constants}")
                    print(f"\n  To fully determine the solution, provide:")
                    
                    # Suggest what's needed
                    order = len(equations)
                    for const in sorted(unsolved_constants, key=str):
                        apostrophes = "'" * order
                        print(f"    {func_name}{apostrophes}(?) = <value>")
                        order += 1
                else:
                    print(f"✓ All constants determined: {const_values}")
                
                # Substitute the constant values into the solution
                new_rhs = sol_expr.subs(const_values)
                new_solution = Eq(solution.lhs, new_rhs)
                
                print(f"\n{'='*70}\n")
                
                return new_solution
            else:
                print("\n✗ Could not solve for constants (inconsistent system?)")
                return solution
                
        except Exception as e:
            print(f"\n✗ Error solving for constants: {e}")
            return solution
    
    def _extract_ic_point(self, ic_str: str, var_name: str) -> float:
        """
        Extract the point value from initial condition string
        
        Args:
            ic_str: Initial condition string
            var_name: Variable name
            
        Returns:
            Float value of the point
        """
        # Remove all apostrophes first to handle derivatives
        cleaned_str = ic_str.replace("'", "")
        
        # Now extract the value inside parentheses
        match = re.search(r'\(([^)]+)\)', cleaned_str)
        
        if not match:
            raise ValueError(f"Could not extract point from '{ic_str}'")
        
        val_str = match.group(1).strip()
        
        # Handle special constants
        if val_str.lower() in ['pi', 'π']:
            return float(pi)
        elif val_str.lower() == 'e':
            return float(E)
        
        # Try to parse as expression
        try:
            val_expr = parse_expr(val_str, transformations=self.transformations)
            return float(val_expr.evalf())
        except:
            try:
                return float(val_str)
            except:
                raise ValueError(f"Could not parse point value: '{val_str}'")
    
    def suggest_initial_conditions(self, result: Dict) -> None:
        """
        Suggest what initial conditions are needed
        
        Args:
            result: Solution result dictionary
        """
        if not result.get('success'):
            return
        
        solution = result['solution']
        if not isinstance(solution, Eq):
            return
        
        x = result['variable']
        sol_expr = solution.rhs
        constants = sol_expr.free_symbols - {x}
        
        if not constants:
            return
        
        num_constants = len(constants)
        func_name = result['function_name']
        
        print(f"\n{'='*70}")
        print(f"INITIAL CONDITIONS NEEDED")
        print(f"{'='*70}")
        print(f"This is a {result['type']} with {num_constants} arbitrary constant(s): {constants}")
        print(f"\nTo get a unique solution, provide {num_constants} initial condition(s):")
        print(f"\nExample format:")
        
        for i in range(num_constants):
            apostrophes = "'" * i
            print(f"  {func_name}{apostrophes}(0) = <value>")
        
        print(f"\nOr use different points:")
        for i in range(min(num_constants, 2)):
            print(f"  {func_name}({i}) = <value>")
        
        print(f"\nSpecial values allowed: pi, e")
        print(f"{'='*70}\n")
    
    def _parse_ode_equation(self, eq_str: str, y, x, func_name: str, var_name: str) -> Eq:
        """
        Parse ODE equation string
        
        Args:
            eq_str: Equation string
            y: Function symbol
            x: Variable symbol
            func_name: Function name
            var_name: Variable name
            
        Returns:
            SymPy equation
        """
        func_pattern = re.escape(func_name)
        
        # Detect maximum derivative order
        max_order = 0
        for i in range(1, min(11, len(eq_str) + 1)):
            if func_name + "'" * i in eq_str:
                max_order = i
        
        # Replace derivatives with SymPy notation
        for order in range(max_order, 0, -1):
            old = func_name + "'" * order
            new = f"Derivative({func_name}({var_name}), {var_name}, {order})"
            eq_str = eq_str.replace(old, new)
        
        # Replace function name with function call
        eq_str = re.sub(rf'\b{func_pattern}\b(?!\()', f'{func_name}({var_name})', eq_str)
        
        # Create local dictionary for parsing
        local_dict = {var_name: x, func_name: y}
        local_dict.update(self._get_safe_constants())
        
        # Parse equation
        if '=' in eq_str:
            parts = eq_str.split('=', 1)
            if len(parts) == 2:
                try:
                    left = parse_expr(parts[0].strip(), local_dict=local_dict,
                                    transformations=self.transformations)
                    right = parse_expr(parts[1].strip(), local_dict=local_dict,
                                     transformations=self.transformations)
                    return Eq(left, right)
                except Exception as e:
                    raise ParseError(f"Failed to parse equation: {str(e)}")
        
        # If no '=', assume equation equals 0
        try:
            expr = parse_expr(eq_str, local_dict=local_dict,
                            transformations=self.transformations)
            return Eq(expr, 0)
        except Exception as e:
            raise ParseError(f"Failed to parse equation: {str(e)}")
    
    def _get_safe_constants(self) -> Dict:
        """
        Get safe mathematical constants
        
        Returns:
            Dictionary of safe constants and functions
        """
        return {
            'exp': exp, 'sin': sin, 'cos': cos, 'tan': tan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'log': log, 'sqrt': sqrt,
            'E': E, 'e': E, 'pi': pi, 'Pi': pi,
            'I': I, 'oo': oo,
            'alpha': self.alpha, 'beta': self.beta,
            'c': self.c, 'k': self.k, 'omega': self.omega,
        }
    
    def _classify_ode(self, equation, func) -> str:
        """
        Classify ODE by order
        
        Args:
            equation: SymPy equation
            func: Function symbol
            
        Returns:
            Classification string
        """
        try:
            order = sp.ode_order(equation, func)
            if order == 1:
                return "First-order ODE"
            elif order == 2:
                return "Second-order ODE"
            elif order == 3:
                return "Third-order ODE"
            else:
                return f"{order}-order ODE"
        except:
            return "ODE"
    
    @timeout(300)
    def solve_pde(self, equation_str: str, function_name: str = 'u',
                  variables: Optional[List[str]] = None,
                  boundary_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Solve a partial differential equation
        
        Args:
            equation_str: PDE as string
            function_name: Name of dependent variable
            variables: List of independent variables
            boundary_conditions: Boundary conditions (not yet implemented)
            
        Returns:
            Dictionary containing solution and metadata
        """
        try:
            equation_str = sanitize_input(equation_str)
            function_name = validate_identifier(function_name)
            
            if variables is None:
                variables = self._detect_pde_variables(equation_str)
            
            var_symbols = [symbols(v, real=True) for v in variables]
            u = Function(function_name)
            
            equation = self._parse_pde_equation(equation_str, u, var_symbols, function_name)
            
            # First, classify the PDE
            pde_type = self._classify_pde(equation, var_symbols, equation_str)
            
            # Then try to get known solution
            known_solution = self._get_known_pde_solution(pde_type, 
                                                           function_name, variables)
            
            if known_solution:
                return {
                    'success': True,
                    'equation': equation,
                    'solution': known_solution['solution'],
                    'solution_text': known_solution['text'],
                    'type': pde_type,
                    'method': 'known_solution',
                    'function': u,
                    'variables': var_symbols,
                    'function_name': function_name,
                    'variable_names': variables,
                    'notes': known_solution.get('notes', ''),
                    'example_solution': known_solution.get('example_solution')
                }
            
            # Try symbolic solution
            solution = None
            solution_method = "symbolic"
            
            try:
                solution = pdsolve(equation, u(*var_symbols))
                solution_method = "pdsolve"
            except NotImplementedError:
                solution_method = "separation_of_variables"
                try:
                    solution = pde_separate(equation, u(*var_symbols), var_symbols)
                except:
                    solution = "Analytical solution exists but cannot be computed symbolically"
                    solution_method = "analytical_form"
            
            return {
                'success': True,
                'equation': equation,
                'solution': solution,
                'type': pde_type,
                'method': solution_method,
                'function': u,
                'variables': var_symbols,
                'function_name': function_name,
                'variable_names': variables
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _get_known_pde_solution(self, pde_type: str, 
                                func_name: str, var_names: List[str]) -> Optional[Dict]:
        """
        Get known solutions for classic PDEs with example solutions
        
        Args:
            pde_type: PDE classification
            func_name: Function name
            var_names: Variable names
            
        Returns:
            Dictionary with solution info or None
        """
        # Heat/Diffusion Equation
        if 'Heat' in pde_type or 'Diffusion' in pde_type:
            if len(var_names) >= 2:
                x, t = var_names[0], var_names[1]
                
                # Example solution function
                def heat_solution(X, T, alpha=0.1, L=1.0, n_terms=10, c=None):
                    """Example heat equation solution"""
                    result = np.zeros_like(X)
                    for n in range(1, n_terms + 1):
                        An = 2.0 * ((-1)**(n+1) + 1) / (n * np.pi)
                        result += An * np.exp(-alpha * (n * np.pi / L)**2 * T) * np.sin(n * np.pi * X / L)
                    return result
                
                return {
                    'solution': f"General solution by separation of variables",
                    'text': f"""
{func_name}({x},{t}) = Sum[A_n * exp(-alpha * n^2 * pi^2 * {t} / L^2) * sin(n * pi * {x} / L), {{n, 1, infinity}}]

where:
  - A_n are Fourier coefficients determined by initial conditions
  - L is the domain length
  - alpha is the thermal diffusivity constant
  - n = 1, 2, 3, ... (infinite series)

Alternative form (fundamental solution):
{func_name}({x},{t}) = (1/sqrt(4*pi*alpha*{t})) * exp(-{x}^2/(4*alpha*{t}))

This represents heat diffusion from a point source.
""",
                    'notes': f'Requires initial condition {func_name}({x},0) and boundary conditions',
                    'example_solution': heat_solution
                }
        
        # Wave Equation
        if 'Wave' in pde_type:
            if len(var_names) >= 2:
                x, t = var_names[0], var_names[1]
                
                # Example solution function
                def wave_solution(X, T, c=1.0, L=1.0, n_terms=10, alpha=None):
                    """Example wave equation solution"""
                    result = np.zeros_like(X)
                    for n in range(1, n_terms + 1):
                        An = 8.0 * L**2 / (n**3 * np.pi**3) * (1 - (-1)**n)
                        result += An * np.cos(n * np.pi * c * T / L) * np.sin(n * np.pi * X / L)
                    return result
                
                return {
                    'solution': f"D'Alembert's solution and separation of variables",
                    'text': f"""
General solution (D'Alembert's formula):
{func_name}({x},{t}) = f({x} - c*{t}) + g({x} + c*{t})

where f and g are arbitrary functions determined by initial conditions.

Alternative form (separation of variables):
{func_name}({x},{t}) = Sum[(A_n * cos(n*pi*c*{t}/L) + B_n * sin(n*pi*c*{t}/L)) * sin(n*pi*{x}/L), {{n, 1, infinity}}]

where:
  - A_n, B_n are coefficients from initial conditions
  - c is the wave speed
  - L is the domain length
  - n = 1, 2, 3, ... (infinite series)
""",
                    'notes': f'Requires initial conditions {func_name}({x},0) and {func_name}_t({x},0)',
                    'example_solution': wave_solution
                }
        
        # Laplace Equation
        if 'Laplace' in pde_type:
            if len(var_names) >= 2:
                x, y = var_names[0], var_names[1]
                
                # Example solution function
                def laplace_solution(X, Y, alpha=None, c=None, L=None, n_terms=None):
                    """Example Laplace equation solution"""
                    return np.sin(np.pi * X) * np.sinh(np.pi * Y)
                
                return {
                    'solution': f"Solution by separation of variables",
                    'text': f"""
General solution (rectangular domain):
{func_name}({x},{y}) = Sum[(A_n * sinh(n*pi*{y}/L) + B_n * cosh(n*pi*{y}/L)) * sin(n*pi*{x}/L), {{n, 1, infinity}}]

Polar coordinates (r, theta):
{func_name}(r,theta) = A_0 + B_0 * ln(r) + Sum[r^n * (A_n * cos(n*theta) + B_n * sin(n*theta)), {{n, 1, infinity}}]
                     + Sum[r^(-n) * (C_n * cos(n*theta) + D_n * sin(n*theta)), {{n, 1, infinity}}]

where coefficients are determined by boundary conditions.
""",
                    'notes': 'Requires boundary conditions on all domain boundaries',
                    'example_solution': laplace_solution
                }
        
        return None
    
    def _parse_pde_equation(self, eq_str: str, u, var_symbols: List, func_name: str) -> Eq:
        """
        Parse PDE equation string
        
        Args:
            eq_str: Equation string
            u: Function symbol
            var_symbols: List of variable symbols
            func_name: Function name
            
        Returns:
            SymPy equation
        """
        local_dict = {str(v): v for v in var_symbols}
        local_dict[func_name] = u
        local_dict.update(self._get_safe_constants())
        
        var_str = ','.join([str(v) for v in var_symbols])
        
        # Replace partial derivatives
        for var in var_symbols:
            var_name = str(var)
            for order in range(5, 0, -1):
                old = f'{func_name}_' + var_name * order
                new = f'Derivative({func_name}({var_str}), {var_name}, {order})'
                eq_str = eq_str.replace(old, new)
        
        # Replace mixed derivatives
        if len(var_symbols) == 2:
            v1, v2 = [str(v) for v in var_symbols]
            eq_str = eq_str.replace(
                f'{func_name}_{v1}{v2}',
                f'Derivative({func_name}({var_str}), {v1}, {v2})'
            )
        
        func_pattern = re.escape(func_name)
        eq_str = re.sub(rf'\b{func_pattern}\b(?!\()', f'{func_name}({var_str})', eq_str)
        
        if '=' in eq_str:
            parts = eq_str.split('=', 1)
            if len(parts) == 2:
                try:
                    left = parse_expr(parts[0].strip(), local_dict=local_dict,
                                    transformations=self.transformations)
                    right = parse_expr(parts[1].strip(), local_dict=local_dict,
                                     transformations=self.transformations)
                    return Eq(left, right)
                except Exception as e:
                    raise ParseError(f"Failed to parse PDE: {str(e)}")
        
        try:
            expr = parse_expr(eq_str, local_dict=local_dict,
                             transformations=self.transformations)
            return Eq(expr, 0)
        except Exception as e:
            raise ParseError(f"Failed to parse PDE: {str(e)}")
    
    def _classify_pde(self, equation, var_symbols: List, equation_str: str) -> str:
        """
        Classify PDE type
        
        Args:
            equation: SymPy equation
            var_symbols: List of variables
            equation_str: Original equation string
            
        Returns:
            Classification string
        """
        eq_str = str(equation)
        orig_str = equation_str.lower()
        
        # Check original string for patterns (more reliable)
        # Wave equation: u_tt = c²*u_xx or u_tt = u_xx
        if ('u_tt' in orig_str or 'utt' in orig_str) and ('u_xx' in orig_str or 'uxx' in orig_str):
            return "Wave Equation (Hyperbolic PDE)"
        
        # Heat equation: u_t = α*u_xx or u_t = u_xx
        if ('u_t' in orig_str and 'u_tt' not in orig_str and 'utt' not in orig_str) and \
           ('u_xx' in orig_str or 'uxx' in orig_str):
            return "Heat/Diffusion Equation (Parabolic PDE)"
        
        # Laplace equation: u_xx + u_yy = 0
        if ('u_xx' in orig_str or 'uxx' in orig_str) and ('u_yy' in orig_str or 'uyy' in orig_str):
            if '= 0' in orig_str or '=0' in orig_str:
                return "Laplace Equation (Elliptic PDE)"
            else:
                return "Poisson Equation (Elliptic PDE)"
        
        # Fallback to derivative analysis
        if 'Derivative' in eq_str:
            if ('t, 2)' in eq_str) and ('x, 2)' in eq_str):
                return "Wave Equation (Hyperbolic PDE)"
            
            if ('t)' in eq_str and 't, 2)' not in eq_str) and ('x, 2)' in eq_str):
                return "Heat/Diffusion Equation (Parabolic PDE)"
            
            if ('x, 2)' in eq_str) and ('y, 2)' in eq_str):
                if eq_str.endswith(', 0)') or 'Eq(0' in eq_str:
                    return "Laplace Equation (Elliptic PDE)"
                else:
                    return "Poisson Equation (Elliptic PDE)"
        
        return "General PDE"
    
    def _detect_pde_variables(self, equation_str: str) -> List[str]:
        """
        Auto-detect PDE variables
        
        Args:
            equation_str: Equation string
            
        Returns:
            List of detected variables
        """
        common_vars = ['x', 'y', 'z', 't']
        detected = []
        
        for var in common_vars:
            if f'_{var}' in equation_str or f'{var}' in equation_str:
                detected.append(var)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for var in detected:
            if var not in seen:
                seen.add(var)
                result.append(var)
        
        if len(result) < 2:
            result = ['x', 't']
        
        return result[:5]
    
    def visualize_ode_solution(self, result: Dict, x_range: Tuple[float, float] = (-5, 5),
                              num_points: int = 1000, save_path: Optional[str] = None,
                              auto_save: bool = True) -> Optional[str]:
        """
        Visualize ODE solution (FIXED - handles Order terms)
        
        Args:
            result: Solution result dictionary
            x_range: Range for x-axis
            num_points: Number of points to plot
            save_path: Path to save plot
            auto_save: Whether to auto-save
            
        Returns:
            Path to saved plot or None
        """
        if not result.get('success'):
            print("Cannot visualize: solving failed")
            return None
        
        try:
            solution = result['solution']
            x_sym = result['variable']
            var_name = result['variable_name']
            func_name = result['function_name']
            
            # Extract expression
            if isinstance(solution, Eq):
                sol_expr = solution.rhs
            else:
                sol_expr = solution
            
            # Remove Order terms (Big-O notation) from series solutions
            if hasattr(sol_expr, 'removeO'):
                sol_expr = sol_expr.removeO()
                print("✓ Removed Order terms from series solution")
            
            # Handle remaining constants
            free_syms = sol_expr.free_symbols - {x_sym}
            if free_syms:
                print(f"\n⚠ Warning: Solution still contains constants: {free_syms}")
                print("  Setting constants to 1 for visualization...")
                for sym in free_syms:
                    sol_expr = sol_expr.subs(sym, 1)
            
            # Create numerical function with multiple fallback strategies
            f = None
            eval_method = "unknown"
            
            # Strategy 1: Standard lambdify
            try:
                f = sp.lambdify(x_sym, sol_expr, modules=['numpy'])
                eval_method = "lambdify (fast)"
            except Exception as e1:
                # Strategy 2: Lambdify with strict=False
                try:
                    from sympy.printing.numpy import NumPyPrinter
                    printer = NumPyPrinter({'strict': False})
                    f = sp.lambdify(x_sym, sol_expr, modules=['numpy'], printer=printer)
                    eval_method = "lambdify non-strict (fast)"
                except Exception as e2:
                    # Strategy 3: Fallback to evalf (slower but always works)
                    print("⚠ Warning: Using slower evaluation method (evalf)")
                    def f(x_val):
                        try:
                            if isinstance(x_val, np.ndarray):
                                return np.array([float(sol_expr.subs(x_sym, float(xv)).evalf()) 
                                               for xv in x_val])
                            else:
                                return float(sol_expr.subs(x_sym, x_val).evalf())
                        except:
                            return np.nan
                    eval_method = "evalf (slow)"
            
            print(f"✓ Using evaluation method: {eval_method}")
            
            # Generate points
            x_vals = np.linspace(x_range[0], x_range[1], num_points)
            y_vals = np.zeros_like(x_vals)
            
            print(f"\nEvaluating solution at {num_points} points...")
            
            # Evaluate with progress indicator
            if eval_method == "evalf (slow)":
                # Slower method - show progress
                for i, xv in enumerate(x_vals):
                    try:
                        result_val = f(xv)
                        if isinstance(result_val, np.ndarray):
                            y_vals[i] = float(result_val.item())
                        else:
                            y_vals[i] = float(result_val)
                    except Exception as e:
                        y_vals[i] = np.nan
                    
                    # Progress indicator
                    if i % (num_points // 10) == 0:
                        print(f"  Progress: {100*i//num_points}%", end='\r')
                print(f"  Progress: 100% - Complete!     ")
            else:
                # Fast method - vectorized
                try:
                    y_vals = f(x_vals)
                    if isinstance(y_vals, (int, float)):
                        y_vals = np.full_like(x_vals, y_vals)
                    print("  ✓ Vectorized evaluation complete")
                except:
                    # Fallback to loop
                    for i, xv in enumerate(x_vals):
                        try:
                            result_val = f(xv)
                            if isinstance(result_val, np.ndarray):
                                y_vals[i] = float(result_val.item())
                            else:
                                y_vals[i] = float(result_val)
                        except:
                            y_vals[i] = np.nan
                    print("  ✓ Loop evaluation complete")
            
            # Handle complex values
            if np.iscomplexobj(y_vals):
                print("⚠ Warning: Solution contains complex values, taking real part")
                y_vals = np.real(y_vals)
            
            # Clip outliers
            finite_mask = np.isfinite(y_vals)
            if not np.any(finite_mask):
                print("✗ Error: No finite values to plot")
                print("  The solution may be undefined in this range")
                return None
            
            num_finite = np.sum(finite_mask)
            if num_finite < num_points * 0.5:
                print(f"⚠ Warning: Only {100*num_finite/num_points:.1f}% of points are finite")
            
            finite_vals = y_vals[finite_mask]
            if len(finite_vals) > 10:
                q1, q99 = np.percentile(finite_vals, [1, 99])
                y_range = q99 - q1
                if y_range > 0:
                    lower = q1 - 3 * y_range
                    upper = q99 + 3 * y_range
                    y_vals = np.clip(y_vals, lower, upper)
            
            # Create plot
            with self._plot_lock:
                fig = plt.figure(figsize=(10, 6), dpi=150)
                ax = fig.add_subplot(111)
                
                # Plot only finite values
                mask = np.isfinite(y_vals)
                ax.plot(x_vals[mask], y_vals[mask], 'b-', linewidth=2, label='Solution')
                
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel(var_name, fontsize=12, fontweight='bold')
                ax.set_ylabel(f"{func_name}({var_name})", fontsize=12, fontweight='bold')
                ax.set_title(f"Solution: {result['type']}", fontsize=14, fontweight='bold')
                ax.legend(loc='best')
                ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
                ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                if save_path is None and auto_save:
                    counter = 1
                    while os.path.exists(f'ode_solution_{counter}.png'):
                        counter += 1
                    save_path = f'ode_solution_{counter}.png'
                
                if save_path:
                    full_path = os.path.abspath(save_path)
                    try:
                        fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        print(f"\n✓ Plot saved to: {save_path}")
                        print(f"  Full path: {full_path}")
                    except Exception as e:
                        print(f"✗ Error saving plot: {e}")
                    finally:
                        plt.close(fig)
                    return full_path
                else:
                    if not self.headless:
                        plt.show()
                    else:
                        print("Running in headless mode - plot not displayed")
                    plt.close(fig)
                    return None
                    
        except Exception as e:
            print(f"✗ Visualization error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def visualize_pde_solution(self, result: Dict, 
                              x_range: Tuple[float, float] = (0, 1),
                              t_range: Tuple[float, float] = (0, 1),
                              num_x_points: int = 100,
                              num_t_points: int = 50,
                              plot_type: str = 'heatmap',
                              save_path: Optional[str] = None,
                              auto_save: bool = True,
                              create_animation: bool = False,
                              animation_frames: int = 50,
                              **kwargs) -> Optional[str]:
        """
        Visualize PDE solution with multiple visualization options
        
        Args:
            result: Solution result dictionary
            x_range: Range for spatial variable
            t_range: Range for time variable
            num_x_points: Number of spatial points
            num_t_points: Number of time points
            plot_type: Type of plot ('heatmap', 'surface', '3d', 'snapshots', 'animation')
            save_path: Path to save plot
            auto_save: Whether to auto-save
            create_animation: Whether to create animation
            animation_frames: Number of animation frames
            **kwargs: Additional parameters (alpha, c, L, etc.)
            
        Returns:
            Path to saved plot/animation or None
        """
        if not result.get('success'):
            print("Cannot visualize: solving failed")
            return None
        
        # Check if we have an example solution function
        example_solution = result.get('example_solution')
        
        if example_solution is None:
            print("\n⚠ No example solution available for visualization")
            print("  PDE visualization requires a specific solution function")
            return None
        
        try:
            var_names = result['variable_names']
            func_name = result['function_name']
            
            # Extract parameters from kwargs
            alpha = kwargs.get('alpha', 0.1)
            c = kwargs.get('c', 1.0)
            L = kwargs.get('L', 1.0)
            n_terms = kwargs.get('n_terms', 20)
            
            print(f"\n{'='*70}")
            print(f"GENERATING PDE VISUALIZATION")
            print(f"{'='*70}")
            print(f"Plot type: {plot_type}")
            print(f"Spatial range: {x_range}")
            print(f"Time range: {t_range}")
            print(f"Grid size: {num_x_points} x {num_t_points}")
            print(f"Parameters: alpha={alpha}, c={c}, L={L}, n_terms={n_terms}")
            print(f"{'='*70}\n")
            
            # Create spatial and temporal grids
            x_vals = np.linspace(x_range[0], x_range[1], num_x_points)
            t_vals = np.linspace(t_range[0], t_range[1], num_t_points)
            X, T = np.meshgrid(x_vals, t_vals)
            
            # Compute solution on grid
            print("Computing solution on grid...")
            U = np.zeros_like(X)
            
            for i in range(num_t_points):
                try:
                    U[i, :] = example_solution(x_vals, t_vals[i], 
                                              alpha=alpha, c=c, L=L, n_terms=n_terms)
                except TypeError:
                    # For Laplace equation (no time dependence)
                    if len(var_names) == 2 and 't' not in var_names:
                        y_vals = np.linspace(t_range[0], t_range[1], num_t_points)
                        Y_grid = np.meshgrid(x_vals, y_vals)[1]
                        U = example_solution(X, Y_grid)
                        break
            
            print("✓ Solution computed successfully")
            
            # Create visualization based on plot_type
            with self._plot_lock:
                if plot_type == 'heatmap':
                    saved_path = self._create_heatmap(X, T, U, result, save_path, auto_save)
                
                elif plot_type == 'surface' or plot_type == '3d':
                    saved_path = self._create_surface_plot(X, T, U, result, save_path, auto_save)
                
                elif plot_type == 'snapshots':
                    saved_path = self._create_snapshots(x_vals, t_vals, U, result, 
                                                       save_path, auto_save)
                
                elif plot_type == 'animation':
                    saved_path = self._create_animation(x_vals, t_vals, U, result,
                                                       save_path, animation_frames)
                
                else:
                    print(f"⚠ Unknown plot type: {plot_type}")
                    print("  Available types: heatmap, surface, 3d, snapshots, animation")
                    return None
                
                return saved_path
                
        except Exception as e:
            print(f"✗ PDE visualization error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_heatmap(self, X, T, U, result, save_path, auto_save):
        """Create 2D heatmap visualization"""
        var_names = result['variable_names']
        func_name = result['function_name']
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        
        im = ax.contourf(X, T, U, levels=50, cmap='RdYlBu_r')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{func_name}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel(var_names[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(var_names[1] if len(var_names) > 1 else 't', 
                     fontsize=12, fontweight='bold')
        ax.set_title(f'{result["type"]} - Heatmap Visualization', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return self._save_plot(fig, save_path, auto_save, 'pde_heatmap')
    
    def _create_surface_plot(self, X, T, U, result, save_path, auto_save):
        """Create 3D surface plot"""
        var_names = result['variable_names']
        func_name = result['function_name']
        
        fig = plt.figure(figsize=(14, 10), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, T, U, cmap='viridis', 
                              linewidth=0, antialiased=True, alpha=0.9)
        
        ax.set_xlabel(var_names[0], fontsize=11, fontweight='bold')
        ax.set_ylabel(var_names[1] if len(var_names) > 1 else 't', 
                     fontsize=11, fontweight='bold')
        ax.set_zlabel(f'{func_name}', fontsize=11, fontweight='bold')
        ax.set_title(f'{result["type"]} - 3D Surface', 
                    fontsize=14, fontweight='bold', pad=20)
        
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label(f'{func_name}', fontsize=10, fontweight='bold')
        
        ax.view_init(elev=25, azim=45)
        
        plt.tight_layout()
        
        return self._save_plot(fig, save_path, auto_save, 'pde_surface')
    
    def _create_snapshots(self, x_vals, t_vals, U, result, save_path, auto_save):
        """Create multiple time snapshots"""
        var_names = result['variable_names']
        func_name = result['function_name']
        
        # Select 6 time snapshots
        num_snapshots = min(6, len(t_vals))
        snapshot_indices = np.linspace(0, len(t_vals)-1, num_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
        axes = axes.flatten()
        
        for idx, t_idx in enumerate(snapshot_indices):
            ax = axes[idx]
            ax.plot(x_vals, U[t_idx, :], 'b-', linewidth=2)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel(var_names[0], fontsize=10, fontweight='bold')
            ax.set_ylabel(f'{func_name}', fontsize=10, fontweight='bold')
            ax.set_title(f't = {t_vals[t_idx]:.3f}', fontsize=11, fontweight='bold')
            ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        
        fig.suptitle(f'{result["type"]} - Time Evolution', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return self._save_plot(fig, save_path, auto_save, 'pde_snapshots')
    
    def _create_animation(self, x_vals, t_vals, U, result, save_path, num_frames):
        """Create animated visualization"""
        var_names = result['variable_names']
        func_name = result['function_name']
        
        print(f"\nCreating animation with {num_frames} frames...")
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Determine y-axis limits
        u_min, u_max = np.min(U), np.max(U)
        u_range = u_max - u_min
        y_limits = (u_min - 0.1*u_range, u_max + 0.1*u_range)
        
        line, = ax.plot([], [], 'b-', linewidth=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_ylim(y_limits)
        ax.set_xlabel(var_names[0], fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{func_name}', fontsize=12, fontweight='bold')
        ax.set_title(f'{result["type"]} - Animation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
        
        # Select frame indices
        frame_indices = np.linspace(0, len(t_vals)-1, num_frames, dtype=int)
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        def animate(frame):
            t_idx = frame_indices[frame]
            line.set_data(x_vals, U[t_idx, :])
            time_text.set_text(f't = {t_vals[t_idx]:.4f}')
            return line, time_text
        
        anim = FuncAnimation(fig, animate, init_func=init,
                           frames=num_frames, interval=50, blit=True)
        
        # Save animation
        if save_path is None:
            counter = 1
            while os.path.exists(f'pde_animation_{counter}.gif'):
                counter += 1
            save_path = f'pde_animation_{counter}.gif'
        
        try:
            writer = PillowWriter(fps=20)
            anim.save(save_path, writer=writer)
            full_path = os.path.abspath(save_path)
            print(f"\n✓ Animation saved to: {save_path}")
            print(f"  Full path: {full_path}")
            print(f"  Frames: {num_frames}")
            print(f"  Duration: ~{num_frames/20:.1f} seconds")
            plt.close(fig)
            return full_path
        except Exception as e:
            print(f"✗ Error saving animation: {e}")
            plt.close(fig)
            return None
    
    def _save_plot(self, fig, save_path, auto_save, default_prefix):
        """Helper method to save plots"""
        if save_path is None and auto_save:
            counter = 1
            while os.path.exists(f'{default_prefix}_{counter}.png'):
                counter += 1
            save_path = f'{default_prefix}_{counter}.png'
        
        if save_path:
            full_path = os.path.abspath(save_path)
            try:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"\n✓ Plot saved to: {save_path}")
                print(f"  Full path: {full_path}")
            except Exception as e:
                print(f"✗ Error saving plot: {e}")
            finally:
                plt.close(fig)
            return full_path
        else:
            if not self.headless:
                plt.show()
            else:
                print("Running in headless mode - plot not displayed")
            plt.close(fig)
            return None
    
    def detect_equation_type(self, equation_str: str) -> str:
        """
        Detect if equation is ODE or PDE
        
        Args:
            equation_str: Equation string
            
        Returns:
            'ode' or 'pde'
        """
        pde_patterns = [
            r'u_(?:x{2,4}|t{2,4}|y{2,4}|z{2,4})',
            r'u_x.{0,20}?u_[tyza]',
            r'[uvw]_(?:x{2,}|t{2,}|y{2,})',
        ]
        
        for pattern in pde_patterns:
            try:
                if re.search(pattern, equation_str):
                    return 'pde'
            except re.error:
                continue
        
        return 'ode'


def parse_initial_conditions(ic_str: str) -> Dict[str, float]:
    """
    Parse initial conditions from string
    
    Args:
        ic_str: Initial conditions string
        
    Returns:
        Dictionary of initial conditions
    """
    if not ic_str or not ic_str.strip():
        return {}
    
    ics = {}
    parts = ic_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '=' not in part:
            continue
        
        key, value = part.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        try:
            ics[key] = float(value)
        except ValueError:
            print(f"⚠ Warning: Could not parse '{part}'")
    
    return ics


def print_header():
    """Print application header"""
    print("=" * 70)
    print("DIFFERENTIAL EQUATION SOLVER")
    print("Version 5.2 - Fixed Visualization with Order Terms")
    print(f"Platform: {platform.system()}")
    print(f"Matplotlib backend: {matplotlib.get_backend()}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 70)


def run_example():
    """Run example problems"""
    print_header()
    
    solver = DifferentialEquationSolver()
    
    print("\n" + "=" * 70)
    print("EXAMPLE 1: First-order ODE with Initial Condition")
    print("=" * 70)
    print("\nEquation: y' = y")
    print("Initial condition: y(0) = 1")
    print("-" * 70)
    
    result = solver.solve_ode("y' = y", initial_conditions={'y(0)': 1})
    
    if result['success']:
        print(f"\n✓ Solution: {result['solution']}")
        print("\nGenerating visualization...")
        solver.visualize_ode_solution(result, save_path='example1_plot.png')
    else:
        print(f"✗ Error: {result['error']}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Heat Equation PDE with Visualization")
    print("=" * 70)
    print("\nEquation: u_t = alpha*u_xx")
    print("-" * 70)
    
    result = solver.solve_pde("u_t = alpha*u_xx")
    
    if result['success']:
        print(f"\n✓ PDE Type: {result['type']}")
        
        # Check if we have solution_text
        if 'solution_text' in result:
            print(f"\n{result['solution_text']}")
        else:
            print(f"\n✓ Solution: {result.get('solution', 'No symbolic solution')}")
        
        # Only try visualization if we have an example solution
        if result.get('example_solution'):
            print("\n" + "-" * 70)
            print("Creating visualizations...")
            print("-" * 70)
            
            # Heatmap
            solver.visualize_pde_solution(result, plot_type='heatmap', 
                                         save_path='heat_eq_heatmap.png',
                                         alpha=0.1, L=1.0)
            
            # 3D Surface
            solver.visualize_pde_solution(result, plot_type='surface',
                                         save_path='heat_eq_surface.png',
                                         alpha=0.1, L=1.0)
            
            # Snapshots
            solver.visualize_pde_solution(result, plot_type='snapshots',
                                         save_path='heat_eq_snapshots.png',
                                         alpha=0.1, L=1.0)
            
            # Animation
            solver.visualize_pde_solution(result, plot_type='animation',
                                         save_path='heat_eq_animation.gif',
                                         alpha=0.1, L=1.0, animation_frames=30)
        else:
            print("\n⚠ No visualization available for this PDE type")
    else:
        print(f"✗ Error: {result['error']}")


def interactive_mode():
    """Interactive mode for solving equations"""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("\nCommands:")
    print("  ode  - Solve an ordinary differential equation")
    print("  pde  - Solve a partial differential equation")
    print("  help - Show help message")
    print("  quit - Exit the program")
    print("\nNote: Plots will be automatically saved to PNG/GIF files")
    
    solver = DifferentialEquationSolver()
    
    while True:
        try:
            print("\n" + "-" * 70)
            command = input("Command: ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif command == 'help':
                print("\n" + "=" * 70)
                print("HELP")
                print("=" * 70)
                print("\nAvailable commands:")
                print("  ode  - Solve an ordinary differential equation")
                print("  pde  - Solve a partial differential equation")
                print("  help - Show this help message")
                print("  quit - Exit the program")
                print("\nODE Examples:")
                print("  y' + y = 0")
                print("  y'' + 4*y = 0")
                print("  y'' - 3*y' + 2*y = 0")
                print("\nPDE Examples:")
                print("  u_t = alpha*u_xx  (Heat equation)")
                print("  u_tt = c**2*u_xx  (Wave equation)")
                print("  u_xx + u_yy = 0   (Laplace equation)")
                print("\nPDE Visualization Types:")
                print("  heatmap   - 2D color map")
                print("  surface   - 3D surface plot")
                print("  snapshots - Multiple time snapshots")
                print("  animation - Animated GIF")
                
            elif command == 'ode':
                print("\n" + "=" * 70)
                print("SOLVE ODE")
                print("=" * 70)
                
                eq_str = input("\nODE: ").strip()
                
                if not eq_str:
                    print("⚠ No equation provided")
                    continue
                
                ic_str = input("Initial conditions (e.g., y(0)=1, y'(0)=0): ").strip()
                
                ics = parse_initial_conditions(ic_str) if ic_str else None
                
                print("\n" + "-" * 70)
                print("Solving...")
                print("-" * 70)
                
                result = solver.solve_ode(eq_str, initial_conditions=ics)
                
                if result['success']:
                    print(f"\n✓ Solution: {result['solution']}")
                    
                    viz = input("\nSave plot? (y/n): ").strip().lower()
                    if viz == 'y':
                        try:
                            x_min = input("x_min (-5): ").strip()
                            x_min = float(x_min) if x_min else -5
                            
                            x_max = input("x_max (5): ").strip()
                            x_max = float(x_max) if x_max else 5
                            
                            filename = input("Filename (press Enter for auto): ").strip()
                            filename = filename if filename else None
                            
                            solver.visualize_ode_solution(
                                result,
                                x_range=(x_min, x_max),
                                save_path=filename
                            )
                        except Exception as e:
                            print(f"✗ Visualization error: {e}")
                else:
                    print(f"\n✗ Error: {result['error']}")
            
            elif command == 'pde':
                print("\n" + "=" * 70)
                print("SOLVE PDE")
                print("=" * 70)
                
                eq_str = input("\nPDE: ").strip()
                
                if not eq_str:
                    print("⚠ No equation provided")
                    continue
                
                vars_str = input("Variables (default x,t): ").strip()
                variables = [v.strip() for v in vars_str.split(',')] if vars_str else None
                
                print("\n" + "-" * 70)
                print("Solving...")
                print("-" * 70)
                
                result = solver.solve_pde(eq_str, variables=variables)
                
                if result['success']:
                    print(f"\n✓ PDE Type: {result['type']}")
                    print(f"  Solution Method: {result['method']}")
                    
                    if result['method'] == 'known_solution' and 'solution_text' in result:
                        print(f"\n{result['solution_text']}")
                        if result.get('notes'):
                            print(f"\nNotes: {result['notes']}")
                        
                        if result.get('example_solution'):
                            viz = input("\nCreate visualization? (y/n): ").strip().lower()
                            if viz == 'y':
                                print("\nVisualization types:")
                                print("  1. Heatmap (2D color map)")
                                print("  2. Surface (3D surface plot)")
                                print("  3. Snapshots (multiple time snapshots)")
                                print("  4. Animation (animated GIF)")
                                print("  5. All of the above")
                                
                                choice = input("\nChoice (1-5): ").strip()
                                
                                try:
                                    alpha = input("Alpha/c parameter (0.1): ").strip()
                                    alpha = float(alpha) if alpha else 0.1
                                    
                                    L = input("Domain length L (1.0): ").strip()
                                    L = float(L) if L else 1.0
                                    
                                    if choice == '1':
                                        solver.visualize_pde_solution(result, plot_type='heatmap',
                                                                     alpha=alpha, L=L)
                                    elif choice == '2':
                                        solver.visualize_pde_solution(result, plot_type='surface',
                                                                     alpha=alpha, L=L)
                                    elif choice == '3':
                                        solver.visualize_pde_solution(result, plot_type='snapshots',
                                                                     alpha=alpha, L=L)
                                    elif choice == '4':
                                        frames = input("Animation frames (30): ").strip()
                                        frames = int(frames) if frames else 30
                                        solver.visualize_pde_solution(result, plot_type='animation',
                                                                     alpha=alpha, L=L,
                                                                     animation_frames=frames)
                                    elif choice == '5':
                                        solver.visualize_pde_solution(result, plot_type='heatmap',
                                                                     alpha=alpha, L=L)
                                        solver.visualize_pde_solution(result, plot_type='surface',
                                                                     alpha=alpha, L=L)
                                        solver.visualize_pde_solution(result, plot_type='snapshots',
                                                                     alpha=alpha, L=L)
                                        solver.visualize_pde_solution(result, plot_type='animation',
                                                                     alpha=alpha, L=L,
                                                                     animation_frames=30)
                                    else:
                                        print("Invalid choice")
                                except Exception as e:
                                    print(f"✗ Visualization error: {e}")
                    else:
                        print(f"\n✓ Solution: {result.get('solution', 'No solution available')}")
                else:
                    print(f"\n✗ Error: {result['error']}")
            
            else:
                print(f"⚠ Unknown command: '{command}'")
                print("  Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user")
            break
        except EOFError:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    """Main entry point"""
    try:
        run_example()
        interactive_mode()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
