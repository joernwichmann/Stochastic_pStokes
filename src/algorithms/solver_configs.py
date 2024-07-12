#use this configuration to display quantities that indicate why the Newton iteration doesn't converge
enable_monitoring={'snes_monitor': None,
            'snes_view': None,
            'ksp_monitor_true_residual': None,
            'snes_converged_reason': None,
            'ksp_converged_reason': None}

enable_light_monitoring={
            'ksp_monitor_true_residual': None,
            'ksp_converged_reason': None}

direct_solve = {"ksp_type": "gmres",
                "mat_type": "aij",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"}

direct_solve_details = {'snes_monitor': None,
           "snes_converged_reason": None,
           'snes_max_it': 120,
           "snes_atol": 1e-8,
           "snes_rtol": 1e-8,
           'snes_linesearch_type': 'nleqerr',
           'ksp_type': 'preonly',
           'pc_type': 'lu', 'mat_type': 'aij',
           "ksp_monitor_true_residual": None,
           "ksp_converged_reason": None,
           'pc_factor_mat_solver_type': 'mumps',
           "mat_mumps_icntl_14": 5000,
           "mat_mumps_icntl_24": 1,
           }