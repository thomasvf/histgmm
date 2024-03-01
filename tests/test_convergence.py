import numpy as np

from histgmm.convergence import ConvergenceTester


def test_convergence_check():
    tol = 0.1
    tol_steps = 2    

    y = np.array([1.0, 0.5, 0.3, 0.25, 0.2, 0.2, 0.15])
    y_diff = np.diff(y)
    converged_at_index = (np.argwhere((np.abs(y_diff) <= tol).cumsum() >= tol_steps)[0] + 1)[0]
    
    tester = ConvergenceTester(tol=tol, tol_steps=tol_steps)
    for idx, value in enumerate(y):
        tester.add_value(value)
        if tester.has_converged():
            print('Converged at step', idx)
            break

    assert idx == converged_at_index



if __name__ == "__main__":
    test_convergence_check()