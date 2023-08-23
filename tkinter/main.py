import tkinter as tk
from tkinter import ttk 

def calcular():
    try:
        num1 = float(entry_num1.get())
        num2 = float(entry_num2.get())
        operacion = combo_operacion.get()

        if operacion == "Suma":
            resultado = num1 + num2
        elif operacion == "Resta":
            resultado = num1 - num2
        elif operacion == "Multiplicación":
            resultado = num1 * num2
        elif operacion == "División":
            resultado = num1 / num2
        else:
            resultado = 0.0

        label_resultado.config(text=f"Resultado: {resultado:.2f}")
    except ValueError:
        label_resultado.config(text="¡Ingresa números válidos!")

# Configuración de la interfaz gráfica
app = tk.Tk()
app.title("Calculadora")
app.geometry("300x250")

label_num1 = tk.Label(app, text="Número 1:")
label_num1.pack()

entry_num1 = tk.Entry(app)
entry_num1.pack()

label_num2 = tk.Label(app, text="Número 2:")
label_num2.pack()

entry_num2 = tk.Entry(app)
entry_num2.pack()

label_operacion = tk.Label(app, text="Operación:")
label_operacion.pack()

operaciones = ["Suma", "Resta", "Multiplicación", "División"]
combo_operacion = ttk.Combobox(app, values=operaciones)
combo_operacion.pack()

btn_calcula = tk.Button(app, text="Calcular", command=calcular)
btn_calcula.pack()

label_resultado = tk.Label(app, text="Resultado:")
label_resultado.pack()

app.mainloop()
