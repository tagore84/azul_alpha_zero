# File: src/players/base_player.py

from abc import ABC, abstractmethod

class BasePlayer(ABC):
    """
    Interfaz base para un jugador de IA de Azul.
    Define el método predict, que toma una observación y retorna una acción válida.
    """

    def __init__(self):
        """
        Inicialización común (semillas, dispositivos, etc.).
        """
        pass

    @abstractmethod
    def predict(self, obs):
        """
        Dado un diccionario de observación `obs`, retorna un índice entero de acción válida.
        """
        raise NotImplementedError