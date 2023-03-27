from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import scipy


class RunsTest:
    """ Clase para implementar la prueba de aleatoriedad basada en rachas. """
    def __init__(self, data):
        """ Constructor de la clase.
        
        :param data: Las mediciones a las que se les aplicará la prueba de aleatoriedad.
        """
        self.data = data
        self._assert_binary_data()
        self.type_i = None
        self.type_ii = None
        self.n_1: int = 0
        self.n_2: int = 0
        self._get_types()
        self.n: int = self.n_1 + self.n_2
        self.r: int = 1
        self._get_runs()
        self.significance_level = 0.0 
    
    def _get_types(self):
        """ Obtenemos los dos tipos de datos.
        Se elige que los objetos del tipo 1 sea de los que menos hay.
        """
        self.type_i, self.type_ii = set(self.data)
        
        for i in self.data:
            if i == self.type_i:
                self.n_1 += 1
            else:
                self.n_2 += 1
        if self.n_1 > self.n_2:
            self.n_1, self.n_2 = self.n_2, self.n_1
            self.type_i, self.type_ii = self.type_ii, self.type_i
        
    def _assert_binary_data(self):
        """ Verifica que los datos sean dicotómicos. """
        assert len(set(self.data)) == 2
        
    def _get_runs(self):
        """ Calcula la cantidad de rachas observadas. """
        cur = self.data[0]
        for i in range(1, self.data.size):
            if cur != self.data[i]:
                self.r += 1
            cur = self.data[i]
            
    def process_non_binary_data(self):
        """ Implementa la opción para pasarle datos no binarios eligiendo un punto de corte para
        volverlos binarios.
        """
        pass
    
    def _test_statistics_asymptotic(self):
        """ Implementa la distribución asintótica de la estadística de prueba.
        """
        pass
    
    def _pmf_even(self, r):
        """ Función de probabilidad de la estadística de prueba cuando r es par.
        
        :param r: Valor que toma R.
        :return: La probabilidad de que R=r
        """
        num = 2 * scipy.special.binom(self.n_1 - 1, r / 2 - 1) * scipy.special.binom(self.n_2 - 1, r / 2 - 1)
        den = scipy.special.binom(self.n, self.n_1)
        return num / den
    
    def _pmf_odd(self, r):
        """ Función de probabilidad de la estadística de prueba cuando r es impar.
        
        :param r: Valor que toma R.
        :return: La probabilidad de que R=r
        """
        num = scipy.special.binom(self.n_1 - 1, (r - 1) / 2) * scipy.special.binom(self.n_2 - 1, (r - 3) / 2)
        num += scipy.special.binom(self.n_1 - 1, (r - 3) / 2) * scipy.special.binom(self.n_2 - 1, (r - 1) / 2)
        den = scipy.special.binom(self.n, self.n_1)
        return num / den
        
    def _test_statistic_exact_two_sized(self, alpha):
        """ Calcula las constantes (cuántiles) de la pmf de R tales que la proba acumulada
        sea menor que el nivel de significancia para la prueba de dos colas.
        
        :param alpha: Nivel de significancia deseado.
        :return: Las constantes que definen la región crítica.
        """
        prob = 0.0
        k_1 = 1
        k_2 = self.n + 1
        while prob <= alpha and (k_1 <= k_2):
            self.significance_level = prob
            k_1 += 1
            k_2 -= 1
            p_1 = self._pmf_even(k_1) if (k_1 % 2 == 0) else self._pmf_odd(k_1)
            p_2 = self._pmf_even(k_2) if (k_2 % 2 == 0) else self._pmf_odd(k_2)
            prob += p_1 + p_2
        k_1 -= 1
        k_2 += 1
        return k_1, k_2
        
    def run_test(self, alpha=0.05, alternative="two-sized", exact=True, cutoff=None):
        """ Realiza la prueba de hipótesis.

        :param alpha: Nivel de significancia.
        :param alternative: Hipótesis alernativa.
        :param exact: Si se hará la prueba exacta.
        :param cutoff: Punto de corte.
        """
        if exact and alternative == "two-sized":
            k_1, k_2 = self._test_statistic_exact_two_sized(alpha)
            print(f"Rechazar H_0 si: R <= {k_1} o si R => {k_2}")
            print(f"Valor que tomó la estadística: R_obs = {self.r}")
            decision = "-> Rechazamos H_0" if (self.r <= k_1 or self.r >= k_2) else "-> No rechazamos H_0"
            print(decision + f" con nivel de significancia {self.significance_level:.3f}")
        elif not exact and alternative == "two-sized":  
            pass
        else:
            pass


class IndependenceTest:
    def __init__(self, table):
        assert isinstance(table, pd.core.frame.DataFrame)
        self.__table = table
        self.__shape = table.shape
        self.__n_dot_i = table.sum(axis=0)
        self.__n_i_dot = table.sum(axis=1)
        self.__n = self.__n_dot_i.sum()
        self.__expected_values = np.zeros(shape=self.__shape)
        self.__compute_expected_values()
    
    def __compute_expected_values(self):
        for i in range(self.__n_i_dot.size):
            for j in range(self.__n_dot_i.size):
                self.__expected_values[i][j] = self.__n_i_dot[i] * self.__n_dot_i[j] / self.__n
        
    def __str__(self):
        text = f"Tabla de contingencia de {self.__shape[0]} x {self.__shape[1]}\n\n"
        text += str(self.__table)
        return text

    def __observed_chi(self):
        jiji = 0
        
        for i in range(self.__shape[0]):
            for j in range(self.__shape[1]):
                jiji += ((self.__table.iloc[i][j] - self.__expected_values[i][j]) ** 2) / self.__expected_values[i][j]
        
        df = (self.__shape[0] - 1) * (self.__shape[1] - 1)

        return jiji, df
    
    def __measures_of_association(self, jiji):
    	""" Calcula las medidas de asociación.
    	
    	:param jiji: La estadística de prueba observada.
    	"""
    	phi = jiji / self.__n
    	rho = np.sqrt(phi ** 2 / (phi ** 2 + 1))
    	t = np.sqrt(phi ** 2 / ((self.__shape[0] - 1)*(self.__shape[1] - 1)))
    	v = np.sqrt(phi ** 2 / (min(self.__shape[0] - 1, self.__shape[1] - 1)))
    	
    	print("\n***** MEDIDAS DE ASOCIACIÓN *****\n")
    	print(f"Medida phi (valores grandes indican mayor asociación): {phi:.4f}")
    	print(f"Coeficiente de contingencia en media cuadrática (0: independientes | 1: completamente asociados): {rho:.4f}")
    	print(f"Índice de T... (valores grandes indican mayor asociación): {t:.4f}")
    	print(f"V de Cramer (valores grandes indican mayor asociación): {v:.4f}")
        
    def run_test(self, alpha):
        jiji, df = self.__observed_chi()
        c_alpha = scipy.stats.chi2.ppf(q=1-alpha, df=df)
        print(f"H_0: 'Las variables son independientes' v.s H_a: 'Las variables están asociadas'")
        print(f"Rechazar H_0 si: jiji > {c_alpha:.3f}")
        print(f"Valor que tomó la estadística jiji: {jiji:.3f}")
        decision = "-> Rechazamos H_0" if jiji > c_alpha else "->No rechazamos H_0"
        print(decision + f" con nivel de significancia {alpha*100}%")
        
        if decision == "-> Rechazamos H_0":
            self.__measures_of_association(jiji)


class BinomialTest(ABC):
    """ Clase para implementar las pruebas binomiales. """
    
    def __init__(self, x):
        """ Constructor de la clase
        
        :param x: Observaciones de la variable aleatoria en cuestión.
        """
        self._x = x
        # Muestra con la cual se realizará la prueba.
        self._sample = None
        # Parámetro p de la distribución Binomial.
        self._p = None
        # Estadística binomial K (valor observado)
        self._k = 0
        # Tamaño de la muestra.
        self._n = x.size
        # Nivel de significancia de la prueba.
        self._significance_level = 0.0
    
    def get_sample(self):
        """ Regresa la muestra a usar (símbolos + y -). 
        
        :return: La muestra a usar para la prueba.
        """
        return self._sample
    
    def get_k(self):
        """ Regresa el valor observado de la estadística de prueba K. """
        return self._k
    
    def _get_c1(self, alpha):
        """ Regresa el valor de la primera constante (cola inferior/izquierda). 
        
        :param alpha: Nivel de significancia.
        :return: El valor de la constante que determina la región crítica.
        """
        c_alpha = 0
        prob = scipy.stats.binom.cdf(c_alpha, self._n, 1 - self._p)
        
        while alpha >= prob:
            c_alpha += 1
            prob = scipy.stats.binom.cdf(c_alpha, self._n, 1 - self._p)
        
        c_alpha -= 1
        self._significance_level = scipy.stats.binom.cdf(c_alpha, self._n, 1 - self._p)
        
        return c_alpha
    
    def _get_c2(self, alpha):
        """ Regresa el valor de la segunda constante (cola superior/derecha). 
        
        :param alpha: Nivel de significancia.
        :return: El valor de la constante que determina la región crítica.
        """
        c_alpha = self._n
        prob = scipy.stats.binom.sf(c_alpha - 1, self._n, 1 - self._p)
        
        while alpha >= prob:
            c_alpha -= 1
            prob = scipy.stats.binom.sf(c_alpha - 1, self._n, 1 - self._p)
        
        c_alpha += 1
        self._significance_level = scipy.stats.binom.sf(c_alpha - 1, self._n, 1 - self._p)
        
        return c_alpha
    
    @abstractmethod
    def run_test(self, alternative, alpha, exact):
        """ Método para ejecutar la prueba de hipótesis.
        
        :param alternative: Hipótesis alternativa.
        :param alpha: Nivel de significancia deseado.
        :param exact: Si se usará la estadística de prueba exacta o no.
        """
        pass


class SignTest(BinomialTest):
    """ Clase para implementar la prueba para el p-ésimo cuantil usando
    la prueba de los signos.
    """
    def __init__(self, x, conditions=None):
        """ Constructor de la clase.
        
        :param x: Observaciones de la variable aleatoria en cuestión.
        :param quantile: El cuantil propuesto para los datos.
        :param p: El orden del cuantil (entre 0 y 1; e.g si p=0.5 para la mediana).
        """
        super().__init__(x)
        self._p = 0.5
        self._conditions = conditions
    
    def __compute_signs(self):
        """ Determina la muestra y calcula la cantidad de signos positivos. """
        self._k = 0 
        signs = []
        
        for i in self._x:
            if self._conditions[0] <= i <= self._conditions[1] or self._conditions[2] <= i <= self._conditions[3]:
                signs.append("+")
                self._k += 1
            else:
                signs.append("-")
        
        self._sample = np.array(signs)
    
    def run_test(self, alternative, alpha=0.10, exact=True):
        """ Ejecuta la prueba de hipótesis.
        
        :param alternative: La hipótesis alternativa.
        :param alpha: Nivel de significancia.
        :param exact: Si se usará la estadística de prueba exacta o no.
        """
        self.__compute_signs()
        
        if exact:
            if alternative == "P('+') != P('-')":
                print("H_0: P('+') = P('-') v.s H_a: P('+') != P('-')")
                c_1 = self._get_c1(alpha/2)
                c_2 = self._get_c2(alpha/2)
                print(f"Rechazar H_0 si K <= {c_1} o K >= {c_2}")
                print(f"Valor que tomó la estadística {self._k}")
                decision = "-> Rechazamos H_0 " if self._k <= c_1 or self._k >= c_2 else "-> No rechazamos H_0 "
                print(decision + f"con nivel de significancia {self._significance_level*100:.2f}%")

            elif alternative == "P('+') < P('-')":
                print("H_0: P('+') = P('-') v.s H_a: P('+') < P('-')")
                c_alpha = self._get_c1(alpha)
                print(f"Rechazar H_0 si K <= {c_alpha}")
                print(f"Valor que tomó la estadística {self._k}")
                decision = "-> Rechazamos H_0 " if self._k <= c_alpha else "-> No Rechazamos H_0 "
                print(decision + f"con nivel de significancia {self._significance_level*100:.2f}%")

            elif alternative == "P('+') > P('-')":
                print("H_0: P('+') = P('-') v.s H_a: P('+') > P('-')")
                c_alpha = self._get_c2(alpha)
                print(f"Rechazar H_0 si K >= {c_alpha}")
                print(f"Valor que tomó la estadística {self._k}")
                decision = "-> Rechazamos H_0 " if self._k >= c_alpha else "-> No rechazamos H_0 "
                print(decision + f"con nivel de significancia {self._significance_level*100:.2f}%")

                
class QuantileSignTest(SignTest):
    """ Clase para implementar la prueba para el p-ésimo cuantil usando
    la prueba de los signos.
    """
    def __init__(self, x, quantile, p):
        """ Constructor de la clase.
        
        :param x: Observaciones de la variable aleatoria en cuestión.
        :param quantile: El cuantil propuesto para los datos.
        :param p: El orden del cuantil (entre 0 y 1; e.g si p=0.5 para la mediana).
        """
        super().__init__(x)
        assert 0 < p < 1
        self._p = p
        self.__quantile = quantile
        self.__compute_signs()
    
    def __compute_signs(self):
        """ Determina la muestra y calcula la cantidad de signos positivos y negativos. """
        signs = []
        
        for i in self._x:
            if i > self.__quantile:
                signs.append("+")
                self._k += 1
            elif i < self.__quantile:
                signs.append("-")
        
        self._sample = np.array(signs)
    
    def run_test(self, alternative, alpha=0.10, exact=True):
        """ Ejecuta la prueba de hipótesis.
        
        :param alternative: La hipótesis alternativa.
                            Opciones:
                                    1. "x_p != n_p0"
                                    2. "x_p < n_p0"
                                    3. "x_p > n_p0"
        :param alpha: Nivel de significancia.
        :param exact: Si se usará la estadística de prueba exacta o no.
        """
        if exact:
            if alternative == "x_p != n_p0":
                print(f"H_0: x_({self._p}) = {self.__quantile} v.s H_a: x_({self._p}) != {self.__quantile}")
                c_1 = self._get_c1(alpha/2)
                c_2 = self._get_c2(alpha/2)
                print(f"Rechazar H_0 si K <= {c_1} o K >= {c_2}")
                print(f"Valor que tomó la estadística {self._k}")
                decision = "-> Rechazamos H_0 " if self._k <= c_1 or self._k >= c_2 else "-> No rechazamos H_0 "
                print(decision + f"con nivel de significancia {self._significance_level*100:.2f}%")

            elif alternative == "x_p < n_p0":
                print(f"H_0: x_({self._p}) = {self.__quantile} v.s H_a: x_({self._p}) < {self.__quantile}")
                c_alpha = self._get_c1(alpha)
                print(f"Rechazar H_0 si K <= {c_alpha}")
                print(f"Valor que tomó la estadística {self._k}")
                decision = "-> Rechazamos H_0 " if self._k <= c_alpha else "-> No Rechazamos H_0 "
                print(decision + f"con nivel de significancia {self._significance_level*100:.2f}%")

            elif alternative == "x_p > n_p0":
                print(f"H_0: x_({self._p}) = {self.__quantile} v.s H_a: x_({self._p}) > {self.__quantile}")
                c_alpha = self._get_c2(alpha)
                print(f"Rechazar H_0 si K >= {c_alpha}")
                print(f"Valor que tomó la estadística {self._k}")
                decision = "-> Rechazamos H_0 " if self._k >= c_alpha  else "-> No rechazamos H_0 "
                print(decision + f"con nivel de significancia {self._significance_level*100:.2f}%")


class EqualityTest(ABC):
    """ Clase para implementar la prueba de igualdad de dos distribuciones (localización). """
    
    def __init__(self, x, y):
        """ Constructor de la clase.
        
        :param x: Mediciones de la primera distribución.
        :param y: Mediciones de la segunda distribución.
        """
        self._x = x
        self._y = y
        self._sample = None
        self._indices = {"X": [], "Y": []}
        self._n_1: int = x.size
        self._n_2: int = y.size
        self._n: int = self._n_1 + self._n_2
        self._significance_level: float = 0.0
        self.__merge_data()
        
    def get_x(self):
        """ Regresa los datos de la primera muestra.
        
        :return: Mediciones de la primera distribución.
        """
        return self._x
    
    def get_y(self):
        """ Regresa los datos de la segunda muestra.
        
        :return: Mediciones de la segunda distribución.
        """
        return self._y
    
    def get_sample(self):
        """ Regresa los datos en una única muestra dicotomizada.
        
        :return: Muestra mezclada.
        """
        return self._sample
    
    def get_indices(self):
        """ Regresa los índices (rangos) de los elementos de la muestra."""
        return self._indices
    
    def __merge_data(self):
        """ Método para dicotomizar los datos.
        
            Se ignoran los empates.
            Se guardan los índices de las X's y de las Y's.
        """
        
        x = np.sort(self._x, kind="mergesort")
        y = np.sort(self._y, kind="mergesort")
        
        merged = []
        
        i, j = 0, 0
        n_1, n_2 = 0, 0
        idx = 0
        
        while (i < self._n_1 and j < self._n_2):
            if x[i] < y[j]:
                merged.append("X")
                self._indices["X"].append(idx)
                n_1 += 1
                i += 1
                idx += 1
            elif x[i] > y[j]:
                merged.append("Y")
                self._indices["Y"].append(idx)
                n_2 += 1
                j += 1
                idx += 1
            else:
                i += 1
                j += 1
        
        while i < self._n_1:
            merged.append("X")
            self._indices["X"].append(idx)
            n_1 += 1
            i += 1
            idx += 1
        
        while j < self._n_2:
            merged.append("Y")
            self._indices["Y"].append(idx)
            n_2 += 1
            j += 1
            idx += 1
        
        self._sample = np.array(merged)
        self._n_1 = n_1
        self._n_2 = n_2
        self._n = self._n_1 + self._n_2


class RunsEqualityTest(EqualityTest):
    """ Clase para implementar la prueba de Wald-Wolfowitz (basada en rachas)."""
    
    def __init__(self, x, y):
        """ Constructor de la clase.
        
        :param x: Mediciones de la primera distribución.
        :param y: Mediciones de la segunda distribución.
        """
        super().__init__(x, y)
        self.__r = 1
        self.__compute_runs()
        
    def get_runs(self):
        """ Regresa la cantidad de rachas en la muestra."""
        return self.__r
        
    def __compute_runs(self):
        """ Calcula la cantidad de rachas observadas."""
        cur = self._sample[0]
        for i in range(1, self._sample.size):
            if cur != self._sample[i]:
                self.__r += 1
            cur = self._sample[i]
    
    def __pmf_even(self, r):
        """ Función de probabilidad de la estadística de prueba cuando r es par.
        
        :param r: Valor que toma R.
        :return: La probabilidad de que R=r
        """
        num = 2 * scipy.special.binom(self._n_1 - 1, r / 2 - 1) * scipy.special.binom(self._n_2 - 1, r / 2 - 1)
        den = scipy.special.binom(self._n, self._n_1)
        return num / den
    
    def __pmf_odd(self, r):
        """ Función de probabilidad de la estadística de prueba cuando r es impar.
        
        :param r: Valor que toma R.
        :return: La probabilidad de que R=r
        """
        num = scipy.special.binom(self._n_1 - 1, (r - 1) / 2) * scipy.special.binom(self._n_2 - 1, (r - 3) / 2)
        num += scipy.special.binom(self._n_1 - 1, (r - 3) / 2) * scipy.special.binom(self._n_2 - 1, (r - 1) / 2)
        den = scipy.special.binom(self._n, self._n_1)
        return num / den
        
    def __test_statistic_exact(self, alpha):
        """ Calcula las constantes (cuántiles) de la pmf de R tales que la proba acumulada
        sea menor que el nivel de significancia para la prueba de dos colas.
        
        :param alpha: Nivel de significancia deseado.
        :return: La constante que define la región crítica.
        """
        prob = 0.0
        k = 1
        while prob <= alpha and (k <= self._n):
            self._significance_level = prob
            k += 1
            p = self.__pmf_even(k) if (k % 2 == 0) else self.__pmf_odd(k)
            prob += p
        k -= 1
        return k
    
    def __test_statistic_asymptotic(self):
        """ Calcula la estadística de prueba asintótica.
        
        :return: El valor que toma la estadística de pruba.
        """
        num = self.__r + 0.5 - 1 - (2 * self._n_1 * self._n_2 / self._n)
        den = 2 * self._n_1 * self._n_2 * (2 * self._n_1 * self._n_2 - self._n)
        den /= (self._n ** 2) * (self._n - 1)
        den = np.sqrt(den)
        
        return num / den
        
    def run_test(self, alpha=0.05, exact=False):
        """ Realiza la prueba de hipótesis. 
        
        :param alpha: El nivel de significancia deseado.
        :param exact: Si la prueba será exacta o asintótica.
        """
        if exact:
            k = self.__test_statistic_exact(alpha)
            print("H_0: F_X = F_Y v.s H_a: F_X != F_Y")
            print(f"Rechazar H_0 si: R <= {k}")
            print(f"Valor que tomó la estadística: R_obs = {self.__r}")
            decision = "-> Rechazamos H_0" if (self.__r <= k) else "-> No rechazamos H_0"
            print(decision + f" con nivel de significancia {self._significance_level*100:.3f}%")
        else:
            z = self.__test_statistic_asymptotic()
            q = -scipy.stats.norm.ppf(q=1-alpha)
            print("H_0: F_X = F_Y v.s H_a: F_X != F_Y")
            print(f"Rechazar H_0 si: Z <= {q:.3f}")
            print(f"Valor que tomó la estadística R: R_obs = {self.__r}")
            print(f"Valor que tomó la estadística Z: Z_obs = {z:.3f}")
            decision = "-> Rechazamos H_0" if z <= q else "->No rechazamos H_0"
            print(decision + f" con nivel de significancia {alpha*100}%")


class MannWhitneyTest(EqualityTest):
    def __init__(self, x, y):
        """ Constructor de la clase.
        
        :param x: Mediciones de la primera distribución.
        :param y: Mediciones de la segunda distribución.
        """
        super().__init__(x, y)
        self.__u: int = 0
        self.__u_prime: int = 0
        self.__compute_u()
        self.__compute_u_prime()
        
    def get_u(self):
        """ Regresa el valor que toma la estadística U. """
        return self.__u
    
    def get_u_prime(self):
        """ Regresa el valor que toma la estadística U'. """
        return self.__u_prime
    
    def __compute_u(self):
        """ Calcula eficientemente el valor de la estadística U. """
        x_indices = self._indices["X"]
        u = sum(x_indices) - (self._n_1 * (self._n_1 - 1) / 2)
        self.__u = int(u)
        
    def __compute_u_prime(self):
        """ Calcula eficientemente el valor de la estadística U'. """
        # Aquí va tu código
    
    def __pmf(self, u):
        """ Función de masa de probabilidad de U (exacta). 
        
        :param u: El valor que toma U.
        """
        # Aquí va tu código
    
    def __test_statistic_asymptotic(self, u):
        """ Calcula la estadística de prueba asintótica.
        
        :param u: El valor que toma U o U'.
        :return: El valor que toma la estadística de pruba.
        """
        z = u + 0.5 - (self._n_1 * self._n_2 / 2)
        z /= np.sqrt((self._n_1 * self._n_2) * (self._n + 1 / 12))
        return z
    
    def run_test(self, alternative, alpha=0.05, exact=False):
        """ Realiza la prueba de hipótesis. 
        
        :param alternative: Hipótesis alternativa.
        :param alpha: Nivel de significancia deseado.
        :param exact: Si la prueba será exacta o asintótica.
        """
        if exact:
            pass
        else:
            if alternative == "F_X != F_Y" or alternative == "F_Y != F_X":
                z = self.__test_statistic_asymptotic(self.__u)
                z_prime = self.__test_statistic_asymptotic(self.__u_prime)
                c_alpha = scipy.stats.norm.ppf(q=alpha/2)
                # Aquí va tu código
                
            elif alternative == "F_X < F_Y" or alternative == "F_Y > F_X":
                z_prime = self.__test_statistic_asymptotic(self.__u_prime)
                c_alpha = scipy.stats.norm.ppf(q=alpha)
                # Aquí va tu código
                
            elif alternative == "F_X > F_Y" or alternative == "F_Y < F_X":
                z = self.__test_statistic_asymptotic(self.__u)
                c_alpha = scipy.stats.norm.ppf(q=alpha)
                print("H_0: F_X = F_Y v.s H_a: F_X > F_Y")
                print(f"Rechazar H_0: F_X = F_Y si: Z <= {c_alpha:.3f}")
                print(f"Valor que tomó la estadística U: U_obs = {self.__u}")
                print(f"Valor que tomó la estadística Z: Z_obs = {z:.3f}")
                decision = "-> Rechazamos H_0" if z <= c_alpha else "->No rechazamos H_0"
                print(decision + f" con nivel de significancia {alpha*100}%")
                

class WilcoxonTest:
    def __init__(self, x, y):
        """ Constructor de la clase.
        
        :param x: Mediciones de la primera distribución.
        :param y: Mediciones de la segunda distribución
        """
        # Aquí va tu código
        
    def __compute_w(self):
        """ Calcula el valor observado de W. """
        # Aquí va tu código
        
    def __test_statistic_asymptotic(self):
        """ Calcula la estadística de prueba asintótica. """
        # Aquí va tu código
        
    def run_test(self, alternative, alpha=0.05, exact=False):
        """ Realiza la prueba de hipótesis.
        
        :param alternative: Hipótesis alternativa.
        :param alpha: Nivel de significancia.
        :param exact: Si la prueba será exacta o asintótica.
        """
        # Aquí va tu código
        
  
