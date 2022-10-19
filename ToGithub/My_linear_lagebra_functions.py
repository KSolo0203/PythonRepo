import numpy as np

# Меняет местами строки/столбцы матрицы:
def SwapAlongAxis(matrix,axis,ind1,ind2):
    result_matrix = matrix.copy()
    if axis == 0:
        result_matrix[ind1] = matrix[ind2]
        result_matrix[ind2] = matrix[ind1]
        return result_matrix
    else:
        result_matrix[...,ind1] = matrix[...,ind2]
        result_matrix[...,ind2] = matrix[...,ind1]
        return result_matrix

# Вычисление минора элемента матрицы:
def Minor(matrix,i,j):
    if matrix.shape[0] != matrix.shape[1]:
        return f"Матрица не квадратная, вычислить детерминант невозможно!"
    else:
        choose = np.arange(len(matrix))
        matrix = matrix[...,choose!=i]
        matrix = matrix[choose!=j]
        return matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]

def IsSquare(matrix):
    return matrix.shape[0] == matrix.shape[1]

# Вычисление алгебраического дополнения элемента матрицы:
def AlgebraicAddition(matrix,i,j):
    if matrix.shape[0] != matrix.shape[1]:
        print(f"Матрица не квадратная, вычислить детерминант невозможно!")
    else:
        choose = np.arange(len(matrix))
        matrix = matrix[choose != i]
        matrix = matrix[...,choose != j]
        if (i + j) % 2 == 0:
            return matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]
        else:
            return (matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]) * -1

# Рекурсивное вычисление детерминанта с использованием теоремы Лапласа. Часть 1.
def AlgebraicAddition(matrix,i,j):
    choose = np.arange(len(matrix))
    matrix = matrix[choose != i]
    matrix = matrix[...,choose != j]
    if matrix.shape[0] > 2:
        if (i + j) % 2 == 0:
            return RecLaplasDeterminant(matrix)
        else:
            return RecLaplasDeterminant(matrix) * -1              
    else:
        if (i + j) % 2 == 0:
            return matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]
        else:
            return (matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]) * -1 

# Рекурсивное вычисление детерминанта с использованием теоремы Лапласа. Часть 2.
def RecLaplasDeterminant(matrix):
    if not IsSquare(matrix):
        print(f"Матрица не квадратная, вычислить детерминант невозможно!")
    ld = 0
    counter = 0
    for element in matrix[0]:
        ld += element * AlgebraicAddition(matrix,0,counter)
        counter += 1
    return ld

# Проверка вырожденности матрицы
def IsNotSingular(matrix):
    return RecLaplasDeterminant(matrix) != 0

# Взятие союзной матрицы
def AdjointMatrix(matrix):
    if IsNotSingular(matrix):
        sideLength = matrix.shape[0]
        B = np.zeros((sideLength,sideLength),dtype=np.float32)
        for i in range(sideLength):
            for j in range(sideLength):
                B[i,j] = AlgebraicAddition(matrix,i,j)
        return B.T
    else:
        return "Вырожденная матрица не имеет союзной самой себе!"

# Взятие обратной матрицы
def InverseMatrix(matrix):
    if IsNotSingular(matrix):
        return AdjointMatrix(matrix) * (1 / RecLaplasDeterminant(matrix))
    else:
        return "Вырожденная матрица не имеет обратной самой себе!"      

# Построение расширенной матрицы
def ExpandedMatrix(matrix,fmColumn):
    if (np.shape(fmColumn)[0] == 1 and np.shape(fmColumn)[1] == np.shape(matrix)[0]):
        return np.hstack((matrix,fmColumn.T))
    elif np.shape(fmColumn)[1] == 1 and np.shape(fmColumn)[0] == np.shape(matrix)[0]:
        return np.hstack((matrix,fmColumn))
    else:
        return f"Ошибка!"

# Проверка вектора-столбца на однородность
def IsHomogenous(fmColumn):
    for member in fmColumn:
        if member.any() != 0:
            return False
        else:
            return True

# Приведение матрицы к ступенчатому виду
def ToStepwise(matrix):
    for j in range(matrix.shape[1]):
        for i in range(1,matrix.shape[0]):
            if i <= j:                
                continue
            elif any(matrix[j]) != 0:
                MakeZero(matrix[i],matrix[j],j)
            else:
                MakeZero(matrix[i],matrix[j-1],j-1)                       
    return matrix 

# Взятие обратной матрицы методом Гаусса (с помощью элементарных преобразований). Основная функция.
def GaussianMatrixInversion(matrix):
    if IsNotSingular(matrix):
        axisLength = np.shape(matrix)[0]
        matrix = np.hstack((matrix,GetE(axisLength)))
        return LeftPartToE(matrix)
    else:
        return "Вырожденная матрица не имеет обратной самой себе!" 

# Построение единичной матрицы n-ого порядка
def GetE(axisLength):
    E = np.zeros((axisLength,axisLength),dtype = np.float32)
    for i in range(axisLength):
        for j in range(axisLength):
            if i == j: E[i,j] = 1 
    return E

# Приведение элемента строки матрицы к нулю путем прибивления к ней другой строки, умноженной на число
def MakeZero(row1,row2,column):
    coeff = row1[column] / row2[column]
    row1 -= row2 * coeff
    return row1

# Приведение элемента строки матрицы к единице путем умножения всех ее элементов на число
def MakeOne(row,column):
    coeff = 1 / row[column]
    row *= coeff
    return row    

# Приведение левой части обединенной матрицы к равносильной единичной матрице
def LeftPartToE(matrix):
    for j in range(int(matrix.shape[1] / 2)):
        for i in range(matrix.shape[0]):
            if i == j:                
                MakeOne(matrix[i],i)
            else:
                MakeZero(matrix[i],matrix[j],j)                       
    return matrix[...,int(matrix.shape[1] / 2):]

# Решение системы линейных уравнений методом Гаусса
def GaussianSolution(matrix):
    solution = []
    for i in range(-1,-np.shape(matrix)[0]-1,-1):
        if any(matrix[i]) != 0:
            solution.append((matrix[i][-1] - ModificatorOfRow(matrix[i],solution)) / matrix[i][i-1])
    return solution[::-1]

# Вспомогательная функция для ешения системы линейных уравнений методом Гаусс
def ModificatorOfRow(row,solution):
    if any(solution):
        modificator = 0
        row = row[::-1]
        for i in range(len(solution)):
            modificator += row[i+1] * solution[i]
        return modificator
    else:
        return 0