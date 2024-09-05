



# IRPF = Impuesto sobre la Renta de las Personas Físicas
def get_IRPF_percentage(salario_bruto):
    if salario_bruto < 12450:
        return 19
    elif salario_bruto < 20200:
        return 24
    elif salario_bruto < 28000:
        return 30
    elif salario_bruto < 35200:
        return 30.30
    elif salario_bruto < 50000:
        return 37.10
    else:
        return 37.20

def calc_salario_neto(salario_bruto):
    return salario_bruto * (1 - get_IRPF_percentage(salario_bruto) / 100)

def get_conyuge_salario_bruto(nro_conyuge):
    while True:
        try:
            conyuge_salario_bruto = float(input(f"Introduzca el salario bruto anual del conyuge {nro_conyuge}: "))
            if conyuge_salario_bruto < 0:
                print("Debe introducir un salario mayor o igual a 0€")
            else :
                return conyuge_salario_bruto
        except ValueError as e:
            print(f"El valor introducido no es válido. {conyuge_salario_bruto} , debe ser un número positivo. Intenta de nuevo.")

def main():
    conyuge_1_salario_bruto = get_conyuge_salario_bruto(1)
    conyuge_2_salario_bruto = get_conyuge_salario_bruto(2)
    sum_salarios_brutos = conyuge_1_salario_bruto + conyuge_2_salario_bruto
    conyuge_1_salario_neto = calc_salario_neto(conyuge_1_salario_bruto)
    conyuge_2_salario_neto = calc_salario_neto(conyuge_2_salario_bruto)
    sum_salarios_netos = conyuge_1_salario_neto + conyuge_2_salario_neto
    print("El salario bruto anual de la pareja es: {:.2f}".format(sum_salarios_brutos))
    print("El salario neto anual de la pareja es: {:.2f}".format(sum_salarios_netos))
    print("El salario neto anual del conyuge 1 es: {:.2f}".format(conyuge_1_salario_neto))
    print("El salario neto anual del conyuge 2 es: {:.2f}".format(conyuge_2_salario_neto))

main()