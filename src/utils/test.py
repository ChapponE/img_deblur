from src.utils.test_utils import available_UnfoldedFB_folder
from src.utils.test_utils import test_models

dataset=[0.5, 7, 0.25]

models = available_UnfoldedFB_folder(notfix=True, fix=True, dataset=dataset)

test_models(models, force_to_save=False, plot=True, ploting_row=True)
