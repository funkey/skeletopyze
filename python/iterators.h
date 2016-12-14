#ifndef SKELETOPYZE_PYTHON_ITERATORS_H__
#define SKELETOPYZE_PYTHON_ITERATORS_H__

#include <imageprocessing/GraphVolume.h>

class GraphVolumeNodeIterator : public std::iterator<std::input_iterator_tag, size_t> {

	const GraphVolume& _graphVolume;
	GraphVolume::Graph::NodeIt _it;

public:

	GraphVolumeNodeIterator(const GraphVolume& g)
		: _graphVolume(g), _it(g.graph()) {}

	GraphVolumeNodeIterator(const GraphVolumeNodeIterator& i)
		: _graphVolume(i._graphVolume), _it(i._it) {}

	GraphVolumeNodeIterator(const GraphVolume& g, const lemon::Invalid& i)
		: _graphVolume(g), _it(i) {}

	GraphVolumeNodeIterator& operator++() {

		++_it;
		return *this;
	}

	GraphVolumeNodeIterator operator++(int) {

		GraphVolumeNodeIterator tmp(*this);
		operator++();
		return tmp;
	}

	bool operator==(const GraphVolumeNodeIterator& rhs) {

		return _it == rhs._it;
	}

	bool operator!=(const GraphVolumeNodeIterator& rhs) {

		return _it != rhs._it;
	}

	size_t operator*() {

		return _graphVolume.graph().id(GraphVolume::Graph::Node(_it));
	}
};

class GraphVolumeEdgeIterator : public std::iterator<std::input_iterator_tag, std::pair<size_t, size_t>> {

	const GraphVolume& _graphVolume;
	GraphVolume::Graph::EdgeIt _it;

public:

	GraphVolumeEdgeIterator(const GraphVolume& g)
		: _graphVolume(g), _it(g.graph()) {}

	GraphVolumeEdgeIterator(const GraphVolumeEdgeIterator& i)
		: _graphVolume(i._graphVolume), _it(i._it) {}

	GraphVolumeEdgeIterator(const GraphVolume& g, const lemon::Invalid& i)
		: _graphVolume(g), _it(i) {}

	GraphVolumeEdgeIterator& operator++() {

		++_it;
		return *this;
	}

	GraphVolumeEdgeIterator operator++(int) {

		GraphVolumeEdgeIterator tmp(*this);
		operator++();
		return tmp;
	}

	bool operator==(const GraphVolumeEdgeIterator& rhs) {

		return _it == rhs._it;
	}

	bool operator!=(const GraphVolumeEdgeIterator& rhs) {

		return _it != rhs._it;
	}

	std::pair<size_t, size_t> operator*() {

		return std::make_pair(
				_graphVolume.graph().id(_graphVolume.graph().u(_it)),
				_graphVolume.graph().id(_graphVolume.graph().v(_it)));
	}
};

class GraphVolumeNodes {

	friend class GraphVolume;

	const GraphVolume& _graphVolume;

public:

	typedef GraphVolumeNodeIterator iterator;
	typedef const GraphVolumeNodeIterator const_iterator;

	GraphVolumeNodes(const GraphVolume& g) : _graphVolume(g) {}

	GraphVolumeNodeIterator begin() const {

		return GraphVolumeNodeIterator(_graphVolume);
	}

	GraphVolumeNodeIterator end() const {

		return GraphVolumeNodeIterator(_graphVolume, lemon::INVALID);
	}

	GraphVolumeNodeIterator cbegin() const {

		return begin();
	}

	GraphVolumeNodeIterator cend() const {

		return end();
	}
};

class GraphVolumeEdges {

	friend class GraphVolume;

	const GraphVolume& _graphVolume;

public:

	GraphVolumeEdges(const GraphVolume& g) : _graphVolume(g) {}

	typedef GraphVolumeEdgeIterator iterator;
	typedef const GraphVolumeEdgeIterator const_iterator;

	GraphVolumeEdgeIterator begin() const {

		return GraphVolumeEdgeIterator(_graphVolume);
	}

	GraphVolumeEdgeIterator end() const {

		return GraphVolumeEdgeIterator(_graphVolume, lemon::INVALID);
	}

	GraphVolumeEdgeIterator cbegin() const {

		return begin();
	}

	GraphVolumeEdgeIterator cend() const {

		return end();
	}
};

#if (defined __clang__ && __clang_major__ < 6) || (defined __GNUC__ && __GNUC__ < 5)
namespace boost { namespace detail {
#else
namespace std {
#endif

template <>
struct iterator_traits<GraphVolumeNodeIterator> {

	typedef size_t value_type;
	typedef size_t reference;
	typedef size_t difference_type;
	typedef typename std::forward_iterator_tag iterator_category;
};

template <>
struct iterator_traits<GraphVolumeEdgeIterator> {

	typedef std::pair<size_t, size_t> value_type;
	typedef std::pair<size_t, size_t> reference;
	typedef size_t difference_type;
	typedef typename std::forward_iterator_tag iterator_category;
};

#if (defined __clang__ && __clang_major__ < 6) || (defined __GNUC__ && __GNUC__ < 5)
}} // namespace boost::detail
#else
} // namespace std
#endif

#endif // SKELETOPYZE_PYTHON_ITERATORS_H__

