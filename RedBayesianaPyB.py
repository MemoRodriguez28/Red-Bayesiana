from pomegranate.distributions import Categorical, ConditionalCategorical
from pomegranate.bayesian_network import BayesianNetwork
from torch.masked import MaskedTensor
from torch import tensor, as_tensor
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module="torch.masked")

# Definir las distribuciones categóricas y condicionales

# Clima (Soleado o Nublado)
Clima = Categorical(
    [[
        0.6,    # Soleado
        0.4,    # Nublado
    ]]
)

# Preferencia por algo frío (Condicional a Clima)
Preferencia = ConditionalCategorical(
    [[
        # Soleado
        [0.8,   # Sí
         0.2],  # No

        # Nublado
        [0.3,   # Sí
         0.7],  # No
    ]]
)

# Elección del Postre (Condicional a Preferencia por algo frío)
Postre = ConditionalCategorical(
    [[
        # Sí (Preferencia por algo frío)
        [0.9,   # Helado
         0.1],  # Tarta

        # No (Preferencia por algo frío)
        [0.2,   # Helado
         0.8],  # Tarta
    ]]
)

# Elección de la Bebida (Condicional a Preferencia por algo frío)
Bebida = ConditionalCategorical(
    [[
        # Sí (Preferencia por algo frío)
        [0.7,   # Refresco
         0.3],  # Agua

        # No (Preferencia por algo frío)
        [0.4,   # Refresco
         0.6],  # Agua
    ]]
)

# Disponibilidad del Postre (Condicional a la Elección del Postre)
Disposicion_postre = ConditionalCategorical(
    [[
        # Helado
        [0.8,   # Disponible
         0.2],  # No disponible

        # Tarta
        [0.6,   # Disponible
         0.4],  # No disponible
    ]]
)

# Disponibilidad de la Bebida (Condicional a la Elección de la Bebida)
Disposicion_bebida = ConditionalCategorical(
    [[
        # Refresco
        [0.9,   # Disponible
         0.1],  # No disponible

        # Agua
        [0.7,   # Disponible
         0.3],  # No disponible
    ]]
)

# Creación de la red bayesiana
modelo = BayesianNetwork()
modelo.add_distributions([Clima, Preferencia, Postre, Bebida, Disposicion_postre, Disposicion_bebida])

# Agregar conexiones
modelo.add_edge(Clima, Preferencia)
modelo.add_edge(Preferencia, Postre)
modelo.add_edge(Preferencia, Bebida)
modelo.add_edge(Postre, Disposicion_postre)
modelo.add_edge(Bebida, Disposicion_bebida)

# Variables lingüísticas
variables = {
    "Clima": ["soleado", "nublado"],
    "Preferencia": ["sí", "no"],
    "Postre": ["helado", "tarta"],
    "Bebida": ["refresco", "agua"],
    "Disposicion_postre": ["disponible", "no disponible"],
    "Disposicion_bebida": ["disponible", "no disponible"]
}

# Probabilidad de un suceso particular
probabilidad = modelo.probability(
    as_tensor(
        [[
            variables["Clima"].index("soleado"),
            variables["Preferencia"].index("sí"),
            variables["Postre"].index("helado"),
            variables["Bebida"].index("refresco"),
            variables["Disposicion_postre"].index("disponible"),
            variables["Disposicion_bebida"].index("disponible"),
        ]]
    )
)
print("\nProbabilidad de un suceso particular:", probabilidad.item())

# Probabilidad de sucesos basados en una observación (por ejemplo, Clima soleado)
observaciones = tensor(
    [[
        variables['Clima'].index("soleado"),  # Clima soleado
        -1,  # Preferencia desconocida
        -1,  # Postre desconocido
        -1,  # Bebida desconocida
        -1,  # disposición del Postre desconocida
        -1,  # disposición de la Bebida desconocida
    ]]
)

# Máscara con True y False de los sucesos observados
observaciones_masked = MaskedTensor(observaciones, mask=(observaciones != -1))

# Predicciones de las variables basadas en la observación
predicciones = modelo.predict_proba(observaciones_masked)

# Imprimir los resultados
print("\nProbabilidad de sucesos basados en una observación:")
for (nombre, valor), pred in zip(variables.items(), predicciones):
    if isinstance(pred, str):
        print(f"{nombre}: {pred}")
    else:
        print(f"{nombre}")
        for valor, prob in zip(valor, pred[0]):
            print(f"    {valor}: {prob:.4f}")
