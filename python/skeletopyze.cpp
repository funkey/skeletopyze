#include <boost/python.hpp>
#include <boost/python/exception_translator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <config.h>
#ifdef HAVE_NUMPY
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

#include <util/exceptions.h>
#include <util/ProgramOptions.h>
#include <imageprocessing/ExplicitVolume.h>
#include <imageprocessing/Skeletonize.h>
#include "logging.h"

template <typename Map, typename K, typename V>
const V& genericGetter(const Map& map, const K& k) { return map[k]; }
template <typename Map, typename K, typename V>
void genericSetter(Map& map, const K& k, const V& value) { map[k] = value; }

template <typename V>
std::vector<V> list_to_vec(boost::python::list l) {
	std::vector<V> vec;
	for (int i = 0; i < boost::python::len(l); ++i) {
		vec.push_back(boost::python::extract<V>(l[i]));
	}
	return vec;
}

template <typename Map, typename K, typename V, typename D>
void featuresSetter(Map& map, const K& k, const V& value) { 
	map.set(k, list_to_vec<D>(value));
}

template <typename Map, typename K, typename V, typename D>
void weightSetter(Map& map, const K& k, const V& value) { 
	map[k] = list_to_vec<D>(value);
}

#if defined __clang__ && __clang_major__ < 6
// std::shared_ptr support
	template<class T> T* get_pointer(std::shared_ptr<T> p){ return p.get(); }
#endif

namespace skeletopyze {

/**
 * Translates an Exception into a python exception.
 *
 **/
void translateException(const Exception& e) {

	if (boost::get_error_info<error_message>(e))
		PyErr_SetString(PyExc_RuntimeError, boost::get_error_info<error_message>(e)->c_str());
	else
		PyErr_SetString(PyExc_RuntimeError, e.what());
}

void init_numpy() {

	std::cout << "Initialize numpy" << std::endl;

	// import_array is a macro expanding to returning different types across 
	// different versions of the NumPy API. This lambda hack works around 
	// that.
	auto imparr = []{ import_array(); };
	imparr();
}

GraphVolume skeletonize_ndarray(PyObject* array) {

	ExplicitVolume<int> volume = volumeFromNumpyArray<int>(array);

	std::cout << "creating graph volume" << std::endl;
	GraphVolume graphVolume(volume);

	std::cout << "creating skeletonizer" << std::endl;
	Skeletonize skeletonizer(graphVolume);

	std::cout << "getting skeleton" << std::endl;
	return skeletonizer.getSkeleton();
}

/**
 * Defines all the python classes in the module libskeletopyze. Here we decide 
 * which functions and data members we wish to expose.
 */
BOOST_PYTHON_MODULE(skeletopyze) {

	boost::python::register_exception_translator<Exception>(&translateException);

	init_numpy();

	// Logging
	boost::python::enum_<logger::LogLevel>("LogLevel")
			.value("Quiet", logger::Quiet)
			.value("Error", logger::Error)
			.value("Debug", logger::Debug)
			.value("All", logger::All)
			.value("User", logger::User)
			;
	boost::python::def("setLogLevel", setLogLevel);
	boost::python::def("getLogLevel", getLogLevel);

	// util::point<float, 3>
	boost::python::class_<util::point<float, 3>>("point_f3")
			.def("x", static_cast<const float&(util::point<float, 3>::*)() const>(&util::point<float, 3>::x),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("y", static_cast<const float&(util::point<float, 3>::*)() const>(&util::point<float, 3>::y),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("z", static_cast<const float&(util::point<float, 3>::*)() const>(&util::point<float, 3>::z),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("__getitem__", &genericGetter<util::point<float,3>, int, float>, boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("__setitem__", &genericSetter<util::point<float,3>, int, float>)
			;

	// util::point<int, 3>
	boost::python::class_<util::point<int, 3>>("point_i3")
			.def("x", static_cast<const int&(util::point<int, 3>::*)() const>(&util::point<int, 3>::x),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("y", static_cast<const int&(util::point<int, 3>::*)() const>(&util::point<int, 3>::y),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("z", static_cast<const int&(util::point<int, 3>::*)() const>(&util::point<int, 3>::z),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("__getitem__", &genericGetter<util::point<int,3>, int, int>, boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("__setitem__", &genericSetter<util::point<int,3>, int, int>)
			;

	// util::box<T, 3> (aka bounding boxes)
	boost::python::class_<util::box<float, 3>>("box_f3")
			.def("min", static_cast<util::point<float, 3>&(util::box<float, 3>::*)()>(&util::box<float, 3>::min),
					boost::python::return_internal_reference<>())
			.def("max", static_cast<util::point<float, 3>&(util::box<float, 3>::*)()>(&util::box<float, 3>::max),
					boost::python::return_internal_reference<>())
			.def("width", &util::box<float, 3>::width)
			.def("height", &util::box<float, 3>::height)
			.def("depth", &util::box<float, 3>::depth)
			.def("contains", &util::box<float, 3>::template contains<float, 3>)
			;
	boost::python::class_<util::box<int, 3>>("box_i3")
			.def("min", static_cast<util::point<int, 3>&(util::box<int, 3>::*)()>(&util::box<int, 3>::min),
					boost::python::return_internal_reference<>())
			.def("max", static_cast<util::point<int, 3>&(util::box<int, 3>::*)()>(&util::box<int, 3>::max),
					boost::python::return_internal_reference<>())
			.def("width", &util::box<int, 3>::width)
			.def("height", &util::box<int, 3>::height)
			.def("depth", &util::box<int, 3>::depth)
			.def("contains", &util::box<int, 3>::template contains<int, 3>)
			;
	boost::python::class_<util::box<unsigned int, 3>>("box_ui3")
			.def("min", static_cast<util::point<unsigned int, 3>&(util::box<unsigned int, 3>::*)()>(&util::box<unsigned int, 3>::min),
					boost::python::return_internal_reference<>())
			.def("max", static_cast<util::point<unsigned int, 3>&(util::box<unsigned int, 3>::*)()>(&util::box<unsigned int, 3>::max),
					boost::python::return_internal_reference<>())
			.def("width", &util::box<unsigned int, 3>::width)
			.def("height", &util::box<unsigned int, 3>::height)
			.def("depth", &util::box<unsigned int, 3>::depth)
			.def("contains", &util::box<unsigned int, 3>::template contains<unsigned int, 3>)
			.def("contains", &util::box<unsigned int, 3>::template contains<int, 3>)
			;

	// ExplicitVolume<int>
	boost::python::class_<ExplicitVolume<int>>("ExplicitVolume_i")
			.def("getBoundingBox", &ExplicitVolume<int>::getBoundingBox, boost::python::return_internal_reference<>())
			.def("getDiscreteBoundingBox", &ExplicitVolume<int>::getDiscreteBoundingBox, boost::python::return_internal_reference<>())
			.def("getResolution", &ExplicitVolume<int>::getResolution, boost::python::return_internal_reference<>())
			.def("cut", &ExplicitVolume<int>::cut)
			.def("__getitem__", &genericGetter<ExplicitVolume<int>,
					util::point<int,3>, int>, boost::python::return_value_policy<boost::python::copy_const_reference>())
			;

	// ExplicitVolume<float>
	boost::python::class_<ExplicitVolume<float>>("ExplicitVolume_f")
			.def("getBoundingBox", &ExplicitVolume<float>::getBoundingBox, boost::python::return_internal_reference<>())
			.def("getDiscreteBoundingBox", &ExplicitVolume<float>::getDiscreteBoundingBox, boost::python::return_internal_reference<>())
			.def("getResolution", &ExplicitVolume<float>::getResolution, boost::python::return_internal_reference<>())
			.def("cut", &ExplicitVolume<float>::cut)
			.def("__getitem__", &genericGetter<ExplicitVolume<float>,
					util::point<float,3>, float>, boost::python::return_value_policy<boost::python::copy_const_reference>())
			;

	// GraphVolume
	boost::python::class_<GraphVolume>("GraphVolume")
			;

	// skeletonize()
	boost::python::def("get_skeleton_graph", skeletonize_ndarray);

	boost::python::def("volumeFromIntNumpyArray", volumeFromNumpyArray<int>);
}

} // namespace skeletopyze
