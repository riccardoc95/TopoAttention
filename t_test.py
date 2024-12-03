from math import sqrt
from scipy.stats import t


def independent_t_test(mean1, std1, n1, mean2, std2, n2, alpha=0.05):
    pooled_variance = (((n1 - 1) * (std1 ** 2)) + ((n2 - 1) * (std2 ** 2))) / (n1 + n2 - 2)

    denominator = sqrt(pooled_variance * (1 / n1 + 1 / n2))
    t_stat = (mean1 - mean2) / denominator

    df = n1 + n2 - 2

    p_value = 2 * (1 - t.cdf(abs(t_stat), df))

    if p_value < alpha:
        conclusion = "Rifiutiamo l'ipotesi nulla: il cambiamento è significativo."
    else:
        conclusion = "Non rifiutiamo l'ipotesi nulla: il cambiamento non è significativo."

    return t_stat, p_value, conclusion



# Esempio di utilizzo
mean1 = 0.0048
std1 = 0.0011
n1 = 3400
mean2 = 0.0049
std2 = 0.0006
n2 = 3400

t_stat, p_value, conclusione = independent_t_test(mean1, std1, n1, mean2, std2, n2)
print(f"T-statistic: {t_stat:.2f}, P-value: {p_value:.4f}")
print(conclusione)
