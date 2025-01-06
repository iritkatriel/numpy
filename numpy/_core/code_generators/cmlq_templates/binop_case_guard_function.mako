${signature}
{
    PyObject *lhs = m1;
    PyObject *rhs = m2;
    /* -- binop_case_guard_function.mako -- */
    int res = (
        (
    %if left_scalar_name is not UNDEFINED:
        Py${left_scalar_name}_CheckExact(lhs) &&
    %else:
        PyArray_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == ${left_numpy_name} &&
    %endif
    %if right_scalar_name is not UNDEFINED:
        Py${right_scalar_name}_CheckExact(rhs)
    %else:
        PyArray_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == ${right_numpy_name}
    %endif
       )
    %if commutative:
    || (
    %if left_scalar_name is not UNDEFINED:
        Py${left_scalar_name}_CheckExact(rhs) &&
    %else:
        PyArray_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == ${left_numpy_name} &&
    %endif
    %if right_scalar_name is not UNDEFINED:
        Py${right_scalar_name}_CheckExact(lhs)
    %else:
        PyArray_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == ${right_numpy_name}
    %endif
       )
    %endif
    );
    return res;
}
