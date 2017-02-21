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
#include "iterators.h"

template <typename Map, typename K, typename V>
const V& genericGetter(const Map& map, const K& k) { return map[k]; }
template <typename Map, typename K, typename V>
void genericSetter(Map& map, const K& k, const V& value) { map[k] = value; }

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

Skeleton skeletonize_ndarray_params_resolution(
		PyObject* array,
		const Skeletonize::Parameters& parameters,
		const util::point<float, 3> resolution) {

	ExplicitVolume<int> volume = volumeFromNumpyArray<int>(array);
	volume.setResolution(resolution.x(), resolution.y(), resolution.z());

	std::cout << "creating graph volume" << std::endl;
	GraphVolume graphVolume(volume);

	std::cout << "creating skeletonizer" << std::endl;
	Skeletonize skeletonizer(graphVolume, parameters);

	std::cout << "getting skeleton" << std::endl;
	return skeletonizer.getSkeleton();
}

Skeleton skeletonize_ndarray_params(PyObject* array, const Skeletonize::Parameters& parameters) {

	return skeletonize_ndarray_params_resolution(array, Skeletonize::Parameters(), util::point<float, 3>(1, 1, 1));
}

Skeleton skeletonize_ndarray(PyObject* array) {

	return skeletonize_ndarray_params(array, Skeletonize::Parameters());
}

SkeletonNodes skeleton_nodes(const Skeleton& skeleton) {

	return SkeletonNodes(skeleton);
}

SkeletonEdges skeleton_edges(const Skeleton& skeleton) {

	return SkeletonEdges(skeleton);
}

util::point<unsigned int,3> skeleton_locations(const Skeleton& skeleton, size_t n) {

	return skeleton.positions()[skeleton.graph().nodeFromId(n)];
}

float skeleton_diameters(const Skeleton& skeleton, size_t n) {

	return skeleton.diameters()[skeleton.graph().nodeFromId(n)];
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

	// util::point<unsinged int, 3>
	boost::python::class_<util::point<unsigned int, 3>>("point_ui3")
			.def("x", static_cast<const unsigned int&(util::point<unsigned int, 3>::*)() const>(&util::point<unsigned int, 3>::x),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("y", static_cast<const unsigned int&(util::point<unsigned int, 3>::*)() const>(&util::point<unsigned int, 3>::y),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("z", static_cast<const unsigned int&(util::point<unsigned int, 3>::*)() const>(&util::point<unsigned int, 3>::z),
					boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("__getitem__", &genericGetter<util::point<unsigned int,3>, unsigned int, unsigned int>, boost::python::return_value_policy<boost::python::copy_const_reference>())
			.def("__setitem__", &genericSetter<util::point<unsigned int,3>, unsigned int, unsigned int>)
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

	// iterators
	boost::python::class_<SkeletonNodes>("SkeletonNodes", boost::python::no_init)
			.def("__iter__", boost::python::iterator<SkeletonNodes>())
			;
	boost::python::class_<SkeletonEdges>("SkeletonEdges", boost::python::no_init)
			.def("__iter__", boost::python::iterator<SkeletonEdges>())
			;

	// "Edges"
	boost::python::class_<std::pair<size_t, size_t>>("SkeletonEdge")
			.def_readwrite("u", &std::pair<size_t, size_t>::first)
			.def_readwrite("v", &std::pair<size_t, size_t>::second)
			;

	// Skeleton
	boost::python::class_<Skeleton>("Skeleton")
			.def("nodes", &skeleton_nodes)
			.def("edges", &skeleton_edges)
			.def("locations", &skeleton_locations)
			.def("diameters", &skeleton_diameters)
			;

	// Skeletonize::Parameters
	boost::python::class_<Skeletonize::Parameters>("Parameters")
			.def_readwrite("min_segment_length", &Skeletonize::Parameters::minSegmentLength)
			.def_readwrite("min_segment_length_ratio", &Skeletonize::Parameters::minSegmentLengthRatio)
			.def_readwrite("skip_explained_nodes", &Skeletonize::Parameters::skipExplainedNodes)
			.def_readwrite("explanation_weight", &Skeletonize::Parameters::explanationWeight)
			.def_readwrite("boundary_weight", &Skeletonize::Parameters::boundaryWeight)
			.def_readwrite("max_num_segments", &Skeletonize::Parameters::maxNumSegments)
			;

	// skeletonize()
	boost::python::def("get_skeleton_graph", skeletonize_ndarray);
	boost::python::def("get_skeleton_graph", skeletonize_ndarray_params);
	boost::python::def("get_skeleton_graph", skeletonize_ndarray_params_resolution);

	boost::python::def("volumeFromIntNumpyArray", volumeFromNumpyArray<int>);
}

} // namespace skeletopyze
