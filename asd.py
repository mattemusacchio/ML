def camino_binario(n):
    pasos = []
    while n > 0:
        if n % 2 == 0:
            pasos.append("izquierda")
            n = n // 2
        else:
            pasos.append("derecha")
            n = (n - 1) // 2
    return pasos[::-1]  # de raíz a hoja

# Ejemplo:
valor_fun7 = int(input("Ingresá el valor devuelto por fun7: "))
camino = camino_binario(valor_fun7)

print("Camino desde la raíz hasta la hoja:")
for paso in camino:
    print(f"→ {paso}")

