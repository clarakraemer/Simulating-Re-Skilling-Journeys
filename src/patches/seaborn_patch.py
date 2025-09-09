import pandas as pd
import seaborn as sns

# Monkey-patch Seaborn to replace 'mode.use_inf_as_null' with 'mode.use_inf_as_na'
original_option_context = pd.option_context

def patched_option_context(*args, **kwargs):
    """Patch pd.option_context to replace deprecated option"""
    args = ['mode.use_inf_as_na' if arg == 'mode.use_inf_as_null' else arg for arg in args]
    return original_option_context(*args, **kwargs)

# Apply the patch
pd.option_context = patched_option_context
