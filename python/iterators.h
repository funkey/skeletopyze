#ifndef SKELETOPYZE_PYTHON_ITERATORS_H__
#define SKELETOPYZE_PYTHON_ITERATORS_H__

#include <imageprocessing/Skeleton.h>

class SkeletonNodeIterator : public std::iterator<std::input_iterator_tag, size_t> {

	const Skeleton& _graphVolume;
	Skeleton::Graph::NodeIt _it;

public:

	SkeletonNodeIterator(const Skeleton& g)
		: _graphVolume(g), _it(g.graph()) {}

	SkeletonNodeIterator(const SkeletonNodeIterator& i)
		: _graphVolume(i._graphVolume), _it(i._it) {}

	SkeletonNodeIterator(const Skeleton& g, const lemon::Invalid& i)
		: _graphVolume(g), _it(i) {}

	SkeletonNodeIterator& operator++() {

		++_it;
		return *this;
	}

	SkeletonNodeIterator operator++(int) {

		SkeletonNodeIterator tmp(*this);
		operator++();
		return tmp;
	}

	bool operator==(const SkeletonNodeIterator& rhs) {

		return _it == rhs._it;
	}

	bool operator!=(const SkeletonNodeIterator& rhs) {

		return _it != rhs._it;
	}

	size_t operator*() {

		return _graphVolume.graph().id(Skeleton::Graph::Node(_it));
	}
};

class SkeletonEdgeIterator : public std::iterator<std::input_iterator_tag, std::pair<size_t, size_t>> {

	const Skeleton& _graphVolume;
	Skeleton::Graph::EdgeIt _it;

public:

	SkeletonEdgeIterator(const Skeleton& g)
		: _graphVolume(g), _it(g.graph()) {}

	SkeletonEdgeIterator(const SkeletonEdgeIterator& i)
		: _graphVolume(i._graphVolume), _it(i._it) {}

	SkeletonEdgeIterator(const Skeleton& g, const lemon::Invalid& i)
		: _graphVolume(g), _it(i) {}

	SkeletonEdgeIterator& operator++() {

		++_it;
		return *this;
	}

	SkeletonEdgeIterator operator++(int) {

		SkeletonEdgeIterator tmp(*this);
		operator++();
		return tmp;
	}

	bool operator==(const SkeletonEdgeIterator& rhs) {

		return _it == rhs._it;
	}

	bool operator!=(const SkeletonEdgeIterator& rhs) {

		return _it != rhs._it;
	}

	std::pair<size_t, size_t> operator*() {

		return std::make_pair(
				_graphVolume.graph().id(_graphVolume.graph().u(_it)),
				_graphVolume.graph().id(_graphVolume.graph().v(_it)));
	}
};

class SkeletonNodes {

	friend class Skeleton;

	const Skeleton& _graphVolume;

public:

	typedef SkeletonNodeIterator iterator;
	typedef const SkeletonNodeIterator const_iterator;

	SkeletonNodes(const Skeleton& g) : _graphVolume(g) {}

	SkeletonNodeIterator begin() const {

		return SkeletonNodeIterator(_graphVolume);
	}

	SkeletonNodeIterator end() const {

		return SkeletonNodeIterator(_graphVolume, lemon::INVALID);
	}

	SkeletonNodeIterator cbegin() const {

		return begin();
	}

	SkeletonNodeIterator cend() const {

		return end();
	}
};

class SkeletonEdges {

	friend class Skeleton;

	const Skeleton& _graphVolume;

public:

	SkeletonEdges(const Skeleton& g) : _graphVolume(g) {}

	typedef SkeletonEdgeIterator iterator;
	typedef const SkeletonEdgeIterator const_iterator;

	SkeletonEdgeIterator begin() const {

		return SkeletonEdgeIterator(_graphVolume);
	}

	SkeletonEdgeIterator end() const {

		return SkeletonEdgeIterator(_graphVolume, lemon::INVALID);
	}

	SkeletonEdgeIterator cbegin() const {

		return begin();
	}

	SkeletonEdgeIterator cend() const {

		return end();
	}
};

#if (defined __clang__ && __clang_major__ < 6) || (defined __GNUC__ && __GNUC__ < 5)
namespace boost { namespace detail {
#else
namespace std {
#endif

template <>
struct iterator_traits<SkeletonNodeIterator> {

	typedef size_t value_type;
	typedef size_t reference;
	typedef size_t difference_type;
	typedef typename std::forward_iterator_tag iterator_category;
};

template <>
struct iterator_traits<SkeletonEdgeIterator> {

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

