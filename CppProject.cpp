#include <iostream>

//float cmult(int int_param, float float_param);

char const *firstMethod() {
    return "This is the first try.";
}

BOOST_PYTHON_MODULE(CppProject) {
    boost::python::def("getTryString", firstMethod); // boost::python is the namespace
}