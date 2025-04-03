from methods.bias_correction.utils import combine_leave_one_out_bias_samples
from methods.bias_correction.apply import apply_bias_correction_to_ensemble


if __name__ == '__main__':
    combine_leave_one_out_bias_samples('nhmv10')
    