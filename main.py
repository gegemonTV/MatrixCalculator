import tkinter as tk
from functools import partial
from numpy import matrix
import numpy as np

main_window = tk.Tk()

main_window.geometry("800x600")
main_window.title("matrix calculator")

matrix_fields = [[tk.Entry(master=main_window)]]
start_matrix = []


def add_column(matrix_entries, master):
    for i in matrix_entries:
        i.append(tk.Entry(master=master))
    for i in range(len(matrix_entries)):
        for j in range(len(matrix_entries[i])):
            matrix_entries[i][j].place(x=30 * j, y=30 + 30 * i, height=30, width=30)


def add_line(matrix_entries, master):
    matrix_entries.append([])
    for i in range(len(matrix_entries[0])):
        matrix_entries[len(matrix_entries) - 1].append(tk.Entry(master=master))
    for i in range(len(matrix_entries)):
        for j in range(len(matrix_entries[i])):
            matrix_entries[i][j].place(x=30 * j, y=30 + 30 * i, height=30, width=30)


def matrix_addition(matrix1, matrix_f):
    matrix2_val = []
    for i in range(len(matrix_f)):
        matrix2_val.append([])
        for j in range(len(matrix_f[i])):
            print(matrix_f[i][j].get())
            matrix2_val[i].append(float(matrix_f[i][j].get()))

    matrix2 = matrix(matrix2_val)

    res_window = tk.Toplevel(main_window)
    res_window.geometry("800x600")

    res_matrix = matrix1 + matrix2

    print(res_matrix.shape[0])

    result = []
    for i in range(res_matrix.shape[0]):
        result.append([])
        for j in range(res_matrix.shape[1]):
            result[i].append(tk.Entry(master=res_window))
            print(res_matrix[i, j])
            result[i][j].insert(0, str(res_matrix[i, j]))

    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j].place(x=30 * j, y=30 * i, width=30, height=30)


def matrix_subtraction(matrix1, matrix_f):
    matrix2_val = []
    for i in range(len(matrix_f)):
        matrix2_val.append([])
        for j in range(len(matrix_f[i])):
            print(matrix_f[i][j].get())
            matrix2_val[i].append(float(matrix_f[i][j].get()))

    matrix2 = matrix(matrix2_val)

    res_window = tk.Toplevel(main_window)
    res_window.geometry("800x600")

    res_matrix = matrix1 - matrix2

    print(res_matrix.shape[0])

    result = []
    for i in range(res_matrix.shape[0]):
        result.append([])
        for j in range(res_matrix.shape[1]):
            result[i].append(tk.Entry(master=res_window))
            print(res_matrix[i, j])
            result[i][j].insert(0, str(res_matrix[i, j]))

    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j].place(x=30 * j, y=30 * i, width=30, height=30)


def matrix_multiplication(matrix1, matrix_f):
    matrix2_val = []
    for i in range(len(matrix_f)):
        matrix2_val.append([])
        for j in range(len(matrix_f[i])):
            print(matrix_f[i][j].get())
            matrix2_val[i].append(float(matrix_f[i][j].get()))

    matrix2 = matrix(matrix2_val)

    res_window = tk.Toplevel(main_window)
    res_window.geometry("800x600")

    print(matrix1.shape, matrix2.shape)
    res_matrix = None
    if matrix1.shape[1] != matrix2.shape[0]:
        matrix2.reshape(matrix1.shape[1], matrix2.shape[0])
        res_matrix = np.matmul(matrix1, matrix2)
    else:
        res_matrix = np.matmul(matrix1, matrix2)

    print(res_matrix.shape[0])

    result = []
    for i in range(res_matrix.shape[0]):
        result.append([])
        for j in range(res_matrix.shape[1]):
            result[i].append(tk.Entry(master=res_window))
            print(res_matrix[i, j])
            result[i][j].insert(0, str(res_matrix[i, j]))

    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j].place(x=30 * j, y=30 * i, width=30, height=30)


def matrix_division(matrix1, matrix_f):
    matrix2_val = []
    for i in range(len(matrix_f)):
        matrix2_val.append([])
        for j in range(len(matrix_f[i])):
            print(matrix_f[i][j].get())
            matrix2_val[i].append(float(matrix_f[i][j].get()))

    matrix2 = matrix(matrix2_val)

    res_window = tk.Toplevel(main_window)
    res_window.geometry("800x600")

    print(matrix1.shape, matrix2.shape)

    try:
        matrix2.reshape(matrix1.shape)
        res_matrix = np.divide(matrix1, matrix2)
        print(res_matrix.shape)

        result = []
        for i in range(res_matrix.shape[0]):
            result.append([])
            for j in range(res_matrix.shape[1]):
                result[i].append(tk.Entry(master=res_window))
                print(res_matrix[i, j])
                result[i][j].insert(0, str(res_matrix[i, j]))

        for i in range(len(result)):
            for j in range(len(result[i])):
                result[i][j].place(x=30 * j, y=30 * i, width=30, height=30)
    except:
        text = tk.Text(master=res_window)
        text.insert(1.0, "Can not divide two matrices because matrices can not be shaped into one form")
        text.pack()


def multiplicate_matrix():
    start_matrix.clear()
    for i in range(len(matrix_fields)):
        start_matrix.append([])
        for j in range(len(matrix_fields[i])):
            print(matrix_fields[i][j].get())
            start_matrix[i].append(float(matrix_fields[i][j].get()))

    matrix1 = matrix(start_matrix)

    print(matrix1)

    add_window = tk.Toplevel(main_window)
    add_window.geometry("800x600")

    mult_fields = [[tk.Entry(master=add_window)]]

    addition_add_col = tk.Button(master=add_window, text="Add column",
                                 command=partial(add_column, mult_fields, add_window))
    addition_add_line = tk.Button(master=add_window, text="Add line",
                                  command=partial(add_line, mult_fields, add_window))
    confirm_button = tk.Button(master=add_window, text="Confirm",
                               command=partial(matrix_multiplication, matrix1, mult_fields))
    addition_add_col.place(x=0, y=0, width=100, height=30)
    addition_add_line.place(x=100, y=0, width=100, height=30)
    confirm_button.place(x=200, y=0, width=100, height=30)

    for l in range(len(mult_fields)):
        for e in range(len(mult_fields[l])):
            mult_fields[l][e].place(x=30 * e, y=30 + 30 * l, height=30, width=30)


def divide_matrix():
    start_matrix.clear()
    for i in range(len(matrix_fields)):
        start_matrix.append([])
        for j in range(len(matrix_fields[i])):
            print(matrix_fields[i][j].get())
            start_matrix[i].append(float(matrix_fields[i][j].get()))

    matrix1 = matrix(start_matrix)

    print(matrix1)

    add_window = tk.Toplevel(main_window)
    add_window.geometry("800x600")

    div_fields = [[tk.Entry(master=add_window)]]

    addition_add_col = tk.Button(master=add_window, text="Add column",
                                 command=partial(add_column, div_fields, add_window))
    addition_add_line = tk.Button(master=add_window, text="Add line", command=partial(add_line, div_fields, add_window))
    confirm_button = tk.Button(master=add_window, text="Confirm", command=partial(matrix_division, matrix1, div_fields))
    addition_add_col.place(x=0, y=0, width=100, height=30)
    addition_add_line.place(x=100, y=0, width=100, height=30)
    confirm_button.place(x=200, y=0, width=100, height=30)

    for l in range(len(div_fields)):
        for e in range(len(div_fields[l])):
            div_fields[l][e].place(x=30 * e, y=30 + 30 * l, height=30, width=30)


def add_matrix():
    start_matrix.clear()
    for i in range(len(matrix_fields)):
        start_matrix.append([])
        for j in range(len(matrix_fields[i])):
            print(matrix_fields[i][j].get())
            start_matrix[i].append(float(matrix_fields[i][j].get()))

    matrix1 = matrix(start_matrix)

    print(matrix1)

    add_window = tk.Toplevel(main_window)
    add_window.geometry("800x600")

    addition_fields = [[tk.Entry(master=add_window)]]

    addition_add_col = tk.Button(master=add_window, text="Add column",
                                 command=partial(add_column, addition_fields, add_window))
    addition_add_line = tk.Button(master=add_window, text="Add line",
                                  command=partial(add_line, addition_fields, add_window))
    confirm_button = tk.Button(master=add_window, text="Confirm",
                               command=partial(matrix_addition, matrix1, addition_fields))
    addition_add_col.place(x=0, y=0, width=100, height=30)
    addition_add_line.place(x=100, y=0, width=100, height=30)
    confirm_button.place(x=200, y=0, width=100, height=30)

    for l in range(len(addition_fields)):
        for e in range(len(addition_fields[l])):
            addition_fields[l][e].place(x=30 * e, y=30 + 30 * l, height=30, width=30)


def subtract_matrix():
    start_matrix.clear()
    for i in range(len(matrix_fields)):
        start_matrix.append([])
        for j in range(len(matrix_fields[i])):
            print(matrix_fields[i][j].get())
            start_matrix[i].append(float(matrix_fields[i][j].get()))

    matrix1 = matrix(start_matrix)

    print(matrix1)

    add_window = tk.Toplevel(main_window)
    add_window.geometry("800x600")

    subtraction_fields = [[tk.Entry(master=add_window)]]

    addition_add_col = tk.Button(master=add_window, text="Add column",
                                 command=partial(add_column, subtraction_fields, add_window))
    addition_add_line = tk.Button(master=add_window, text="Add line",
                                  command=partial(add_line, subtraction_fields, add_window))
    confirm_button = tk.Button(master=add_window, text="Confirm",
                               command=partial(matrix_subtraction, matrix1, subtraction_fields))
    addition_add_col.place(x=0, y=0, width=100, height=30)
    addition_add_line.place(x=100, y=0, width=100, height=30)
    confirm_button.place(x=200, y=0, width=100, height=30)

    for l in range(len(subtraction_fields)):
        for e in range(len(subtraction_fields[l])):
            subtraction_fields[l][e].place(x=30 * e, y=30 + 30 * l, height=30, width=30)


def determine_matrix():
    start_matrix.clear()
    for i in range(len(matrix_fields)):
        start_matrix.append([])
        for j in range(len(matrix_fields[i])):
            print(matrix_fields[i][j].get())
            start_matrix[i].append(float(matrix_fields[i][j].get()))

    matrix1 = matrix(start_matrix)
    print(np.linalg.det(matrix1))

    det_win= tk.Toplevel(main_window)
    text = tk.Text(master=det_win)
    text.insert(1.0, f"det = {np.linalg.det(matrix1)}")
    if np.linalg.det(matrix1) == 0:
        text.insert(2.0, "\nвырожденная")
    else:
        text.insert(2.0, "\nневырожденная")
    text.pack()


matrix_add_column = tk.Button(text="Add column", command=partial(add_column, matrix_fields, main_window))
matrix_add_line = tk.Button(text="Add line", command=partial(add_line, matrix_fields, main_window))

matrix_add = tk.Button(text="Addition", command=add_matrix)
matrix_subtract = tk.Button(text="Subtraction", command=subtract_matrix)
matrix_multiply = tk.Button(text="Multiplication", command=multiplicate_matrix)
matrix_divide = tk.Button(text="Division", command=divide_matrix)
matrix_determine = tk.Button(text="Determine", command=determine_matrix)

matrix_add_column.place(x=0, y=0, width=100, height=30)
matrix_add_line.place(x=100, y=0, width=100, height=30)
matrix_add.place(x=300, y=0, width=100, height=30)
matrix_subtract.place(x=400, y=0, width=100, height=30)
matrix_multiply.place(x=500, y=0, width=100, height=30)
matrix_divide.place(x=600, y=0, width=100, height=30)
matrix_determine.place(x=700, y=0, width=100, height=30)

for l in range(len(matrix_fields)):
    for e in range(len(matrix_fields[l])):
        matrix_fields[l][e].place(x=30 * e, y=30 + 30 * l, height=30, width=30)
        # matrix_fields[l][e].pack()

if __name__ == "__main__":
    main_window.mainloop()
