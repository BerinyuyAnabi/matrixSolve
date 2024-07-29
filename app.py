#Importing the necessary libraries 
from flask import Flask, request, render_template
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64

app = Flask("MatrixSolver")

def createMatrix(row, column, values):
    ''' A method that creates a matrix with a given dimension.
  Arguments: row and column represent the number of rows and columns respectively.
  Returns a standard matrix
  '''
  # Initialize an empty matrix with correct dimensions
  # Taking input for the matrix elements while Iterating over the rows and coolumns of the matrix
    matrix = [[values[i * column + j] for j in range(column)] for i in range(row)]
    return sp.Matrix(matrix)

def solveMatrix(matrix):
    ''' A method that takes the augmented matrix as input and solves it. '''
    rref, pivot_columns = matrix.rref()
    return rref

def augment_matrix(standard_matrix, b):
    ''' A function that takes a standard matrix and augments it with a vector b. '''                                            
    b = sp.Matrix(b)
    augmented_matrix = standard_matrix.row_join(b)
    return augmented_matrix

def plot_solution_sets(solution_set1, solution_set2):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(solution_set1) == 1:
        t = np.linspace(-10, 10, 100)
        null_vector = np.array(solution_set1[0]).astype(np.float64).flatten()
        # null_vector = solution_set1[:, 0]

        plotting_points = []
        for i in range(3):
            plotting_points.append(t * null_vector[i])
        plotting_points = np.array(plotting_points)
        ax.plot(plotting_points[0], plotting_points[1], plotting_points[2], label='Null Space (Ax = 0)')
    
    elif len(solution_set1) == 2:
        t1, t2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
        Z = solution_set1[0][0] * t1 + solution_set1[1][0] * t2
        ax.plot_surface(t1, t2, Z, alpha=0.9)
    
    if solution_set2 is not None and len(solution_set2) >= 3:
        ax.scatter(solution_set2[0], solution_set2[1], solution_set2[2], color='r', label='Heterogeneous Solution Set (Ax = b)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='best')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    return plot_url


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        rows = int(request.form['rows'])
        columns = int(request.form['columns'])
        values = list(map(int, request.form.getlist('matrix_values')))
        b_values = list(map(int, request.form.getlist('b_values')))

        # Create matrix and solve it
        matrix = createMatrix(rows, columns, values)
        b = sp.Matrix(b_values)
        d = sp.zeros(rows, 1)

        # Augment matrices
        non_homogeneous_matrix = augment_matrix(matrix, b)
        homogeneous_matrix = augment_matrix(matrix, d)

        # Solve matrices
        rref_non_homogeneous = solveMatrix(non_homogeneous_matrix)
        rref_homogeneous = solveMatrix(homogeneous_matrix)

        # Convert to numpy for further calculations
        matrix_np = np.array(matrix.tolist(), dtype=float)
        solution_set1 = matrix.nullspace()  # finding the nullspace
        b_np = np.array(b_values).reshape(-1, 1).astype(np.float64)

        # Determine ranks
        rank_A = np.linalg.matrix_rank(matrix_np)
        augmented_matrix = np.hstack((matrix_np, b_np))
        rank_augmented = np.linalg.matrix_rank(augmented_matrix)

        solution_set2 = None
        if rank_A == rank_augmented:
            solution_set2 = np.linalg.lstsq(matrix_np, b_np, rcond=None)[0]

        # Generate plot
        plot_url = plot_solution_sets(solution_set1, solution_set2)

        # Convert matrices to formatted strings for rendering
        def format_matrix(matrix):
            return '\n'.join(['\t'.join(map(str, row)) for row in matrix.tolist()])

        input_matrix_str = format_matrix(matrix)
        non_homogeneous_matrix_str = format_matrix(non_homogeneous_matrix)
        homogeneous_matrix_str = format_matrix(homogeneous_matrix)
        rref_non_homogeneous_str = format_matrix(rref_non_homogeneous)
        rref_homogeneous_str = format_matrix(rref_homogeneous)
        solution_set1_str = '\n'.join(['\t'.join(map(str, s.tolist())) for s in solution_set1])
        solution_set2_str = '\n'.join(map(str, solution_set2.tolist())) if solution_set2 is not None else None

        # Render template with results
        return render_template('index.html', 
                               input_matrix=input_matrix_str,
                               non_homogeneous_matrix=non_homogeneous_matrix_str,
                               homogeneous_matrix=homogeneous_matrix_str,
                               rref_non_homogeneous=rref_non_homogeneous_str,
                               rref_homogeneous=rref_homogeneous_str,
                               solution_set1=solution_set1_str,
                               solution_set2=solution_set2_str,
                               plot_url=plot_url)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
