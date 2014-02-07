import numpy as np

unsigned_integer_types = [np.uint, np.uint8, np.uint16, np.uint32, np.uint64]
signed_integer_types = [np.int, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64]
integer_types = unsigned_integer_types + signed_integer_types

real_types = [np.float, np.float_, np.float16, np.float32, np.float64, np.float128]

complex_types = [np.complex, np.complex_, np.complex64, np.complex128, np.complex256]

numeric_types = integer_types + real_types + complex_types
allowed_types = numeric_types
