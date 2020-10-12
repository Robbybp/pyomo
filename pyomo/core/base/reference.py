#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyutilib.misc import flatten_tuple
from pyomo.common import DeveloperError
from pyomo.core.base.set import SetOf, _SetDataBase
from pyomo.core.base.component import Component, ComponentData
from pyomo.core.base.indexed_component import (
    IndexedComponent, UnindexedComponent_set
)
from pyomo.core.base.indexed_component_slice import (
    IndexedComponent_slice, _IndexedComponent_slice_iter
)

import six
from six import iteritems, advance_iterator

if six.PY3:
    from collections.abc import MutableMapping as collections_MutableMapping
    from collections.abc import Set as collections_Set
else:
    from collections import MutableMapping as collections_MutableMapping
    from collections import Set as collections_Set

_NotSpecified = object()

class _fill_in_known_wildcards(object):
    """Variant of "six.advance_iterator" that substitutes wildcard values

    This object is initialized with a tuple of index values.  Calling
    the resulting object on a :py:class:`_slice_generator` will
    "advance" the iterator, substituting values from the tuple into the
    slice wildcards (":" indices), and returning the resulting object.
    The motivation for implementing this as an iterator is so that we
    can re-use all the logic from
    :py:meth:`_IndexedComponent_slice_iter.__next__` when looking up
    specific indices within the slice.

    Parameters
    ----------
    wildcard_values : tuple of index values
        a tuple containing index values to substitute into the slice wildcards

    look_in_index : :py:class:`bool` [optional]
        If True, the iterator will also look for matches using the
        components' underlying index_set() in addition to the (sparse)
        indices matched by the components' __contains__()
        method. [default: False]

    get_if_not_present : :py:class:`bool` [optional]
        If True, the iterator will attempt to retrieve data objects
        (through getitem) for indexes that match the underlying
        component index_set() but do not appear in the (sparse) indices
        matched by __contains__.  get_if_not_present implies
        look_in_index.  [default: False]

    """
    def __init__(self, wildcard_values,
                 look_in_index=False,
                 get_if_not_present=False):
        self.base_key = wildcard_values
        self.key = list(wildcard_values)
        self.known_slices = set()
        self.look_in_index = look_in_index or get_if_not_present
        self.get_if_not_present = get_if_not_present

    def __call__(self, _slice):
        """Advance the specified slice generator, substituting wildcard values

        This advances the passed :py:class:`_slice_generator
        <pyomo.core.base.indexed_component_slice._slice_generator>` by
        substituting values from the `wildcard_values` list for any
        wildcard slices ("`:`").

        Parameters
        ----------
        _slice : pyomo.core.base.indexed_component_slice._slice_generator
            the slice to advance
        """
        if _slice in self.known_slices:
            # Why does iteration stop when a "known slice" is encountered?
            # I.e. when a slice is encountered for the second time?
            # Note that _slice is actually a slice_generator...
            # We only want to fill in the wildcards for each slice once.
            # Why would we ever encounter the same slice twice?
            # 
            # known_slices contains all the slice_generators that have
            # previously been encountered using this iterator
            raise StopIteration()
        self.known_slices.add(_slice)

        # What is the purpose of `idx_count`?
        if _slice.ellipsis is None:
            idx_count = _slice.explicit_index_count
        elif not _slice.component.is_indexed():
            idx_count = 1
        else:
            idx_count = _slice.component.index_set().dimen
            if idx_count is None:
                raise SliceEllipsisLookupError(
                    "Cannot lookup elements in a _ReferenceDict when the "
                    "underlying slice object contains ellipsis over a jagged "
                    "(dimen=None) Set")
        try:
            # Here we assemble the index we will actually use to access
            # the component.
            idx = tuple(
                _slice.fixed[i] if i in _slice.fixed else self.key.pop(0)
                for i in range(idx_count))
            # _slice corresponds to some sliced entry in the call/iter stacks
            # that contains the information describing the slice.
            # Here we fill in an index with the fixed indices from the slice
            # the wildcard indices from the provided key.
        except IndexError:
            # Occurs if we try to pop from an empty list.
            raise KeyError(
                "Insufficient values for slice of indexed component '%s' "
                "(found evaluating slice index %s)"
                % (_slice.component.name, self.base_key))

        if idx in _slice.component:
            # Set the last_index of the slice generator to the total
            # index. Unclear why this is important.
            _slice.last_index = idx
            # Return the component. We have successfully accessed
            # the data value.
            #print(_slice.component[idx])
            return _slice.component[idx]
        elif len(idx) == 1 and idx[0] in _slice.component:
            # `idx` is a len-1 tuple. It is the scalar, not the tuple,
            # that is contained by the component.
            _slice.last_index = idx
            #print(_slice.component[idx])
            return _slice.component[idx[0]]
        elif self.look_in_index:
            if idx in _slice.component.index_set():
                _slice.last_index = idx
                return _slice.component[idx] if self.get_if_not_present \
                    else None
            elif len(idx) == 1 and idx[0] in _slice.component.index_set():
                _slice.last_index = idx
                return _slice.component[idx[0]] if self.get_if_not_present \
                    else None

        raise KeyError(
            "Index %s is not valid for indexed component '%s' "
            "(found evaluating slice index %s)"
            % (idx, _slice.component.name, self.base_key))

    def check_complete(self):
        if self.key:
            raise KeyError("Extra (unused) values for slice index %s"
                           % ( self.base_key, ))


class SliceEllipsisLookupError(Exception):
    pass

class _ReferenceDict(collections_MutableMapping):
    """A dict-like object whose values are defined by a slice.

    This implements a dict-like object whose keys and values are defined
    by a component slice (:py:class:`IndexedComponent_slice`).  The
    intent behind this object is to replace the normal ``_data``
    :py:class:`dict` in :py:class:`IndexedComponent` containers to
    create "reference" components.

    Parameters
    ----------
    component_slice : :py:class:`IndexedComponent_slice`
        The slice object that defines the "members" of this mutable mapping.
    """
    def __init__(self, component_slice):
        self._slice = component_slice

    def __contains__(self, key):
        try:
            advance_iterator(self._get_iter(self._slice, key))
            return True
        except (StopIteration, KeyError):
            return False
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            # Brute force (linear time) lookup
            _iter = iter(self._slice)
            for item in _iter:
                if _iter.get_last_index_wildcards() == key:
                    return True
            return False

    def __getitem__(self, key):
        try:
            return advance_iterator(
                self._get_iter(self._slice, key, get_if_not_present=True)
            )
        except StopIteration:
            raise KeyError("KeyError: %s" % (key,))
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            # Brute force (linear time) lookup
            _iter = iter(self._slice)
            for item in _iter:
                # Why could this help us in the case of a
                # SliceEllipsisLookupError?
                if _iter.get_last_index_wildcards() == key:
                    return item
            raise KeyError("KeyError: %s" % (key,))

    def __setitem__(self, key, val):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == IndexedComponent_slice.get_item:
            tmp._call_stack[-1] = (
                IndexedComponent_slice.set_item,
                tmp._call_stack[-1][1],
                val )
        elif op == IndexedComponent_slice.slice_info:
            tmp._call_stack[-1] = (
                IndexedComponent_slice.set_item,
                tmp._call_stack[-1][1],
                val )
        elif op == IndexedComponent_slice.get_attribute:
            tmp._call_stack[-1] = (
                IndexedComponent_slice.set_attribute,
                tmp._call_stack[-1][1],
                val )
        else:
            raise DeveloperError(
                "Unexpected slice _call_stack operation: %s" % op)
        try:
            advance_iterator(self._get_iter(tmp, key, get_if_not_present=True))
        except StopIteration:
            pass

    def __delitem__(self, key):
        tmp = self._slice.duplicate()
        op = tmp._call_stack[-1][0]
        if op == IndexedComponent_slice.get_item:
            # If the last attribute of the slice gets an item,
            # change it to delete the item
            tmp._call_stack[-1] = (
                IndexedComponent_slice.del_item,
                tmp._call_stack[-1][1] )
        elif op == IndexedComponent_slice.slice_info:
            assert len(tmp._call_stack) == 1
            _iter = self._get_iter(tmp, key)
            try:
                advance_iterator(_iter)
                del _iter._iter_stack[0].component[_iter.get_last_index()]
                return
            except StopIteration:
                raise KeyError("KeyError: %s" % (key,))
        elif op == IndexedComponent_slice.get_attribute:
            # If the last attribute of the slice retrieves an attribute,
            # change it to delete the attribute
            tmp._call_stack[-1] = (
                IndexedComponent_slice.del_attribute,
                tmp._call_stack[-1][1] )
        else:
            raise DeveloperError(
                "Unexpected slice _call_stack operation: %s" % op)
        try:
            advance_iterator(self._get_iter(tmp, key))
        except StopIteration:
            pass

    def __iter__(self):
        return self._slice.wildcard_keys()

    def __len__(self):
        return sum(1 for i in self._slice)

    def iteritems(self):
        """Return the wildcard, value tuples for this ReferenceDict

        This method is necessary because the default implementation
        iterates over the keys and looks the values up in the
        dictionary.  Unfortunately some slices have structures that make
        looking up components by the wildcard keys very expensive
        (linear time; e.g., the use of elipses with jagged sets).  By
        implementing this method without using lookups, general methods
        that iterate over everything (like component.pprint()) will
        still be linear and not quadratic time.

        """
        return self._slice.wildcard_items()

    def itervalues(self):
        """Return the values for this ReferenceDict

        This method is necessary because the default implementation
        iterates over the keys and looks the values up in the
        dictionary.  Unfortunately some slices have structures that make
        looking up components by the wildcard keys very expensive
        (linear time; e.g., the use of ellipses with jagged sets).  By
        implementing this method without using lookups, general methods
        that iterate over everything (like component.pprint()) will
        still be linear and not quadratic time.

        """
        return iter(self._slice)

    def _get_iter(self, _slice, key, get_if_not_present=False):
        if key.__class__ not in (tuple, list):
            key = (key,)
        return _IndexedComponent_slice_iter(
            _slice,
            _fill_in_known_wildcards(flatten_tuple(key),
                                     get_if_not_present=get_if_not_present)
        )
        # Does using this iterator in __getitem__ mean that looking up an
        # index is linear in the size of the "indexing set?"
        #
        # Why is an IC_slice_iter object used here?

if six.PY3:
    _ReferenceDict.items = _ReferenceDict.iteritems
    _ReferenceDict.values = _ReferenceDict.itervalues

class _ReferenceSet(collections_Set):
    """A set-like object whose values are defined by a slice.

    This implements a dict-like object whose members are defined by a
    component slice (:py:class:`IndexedComponent_slice`).
    :py:class:`_ReferenceSet` differs from the
    :py:class:`_ReferenceDict` above in that it looks in the underlying
    component ``index_set()`` for values that match the slice, and not
    just the (sparse) indices defined by the slice.

    Parameters
    ----------
    component_slice : :py:class:`IndexedComponent_slice`
        The slice object that defines the "members" of this set

    """
    def __init__(self, component_slice):
        self._slice = component_slice

    def __contains__(self, key):
        try:
            advance_iterator(self._get_iter(self._slice, key))
            return True
        except (StopIteration, KeyError):
            return False
        except SliceEllipsisLookupError:
            if type(key) is tuple and len(key) == 1:
                key = key[0]
            # Brute force (linear time) lookup
            _iter = iter(self._slice)
            for item in _iter:
                if _iter.get_last_index_wildcards() == key:
                    return True
            return False

    def __iter__(self):
        return self._slice.index_wildcard_keys()

    def __len__(self):
        return sum(1 for _ in self)

    def _get_iter(self, _slice, key):
        if key.__class__ not in (tuple, list):
            key = (key,)
        return _IndexedComponent_slice_iter(
            _slice,
            _fill_in_known_wildcards(flatten_tuple(key), look_in_index=True),
            iter_over_index=True
        )


def _identify_wildcard_sets(iter_stack, index):
    # if we have already decided that there isn't a common index for the
    # slices, there is nothing more we can do.  Bail.
    if index is None:
        return index

    # Walk the iter_stack that led to the current item and try to
    # identify the component wildcard sets
    #
    # `tmp` will be a list of dicts mirroring the iter_stack. Each
    # dict maps position within that level's component's "subsets"
    # list to the set at that position if it is a wildcard set.
    tmp = [None]*len(iter_stack)
    for i, level in enumerate(iter_stack):
        if level is not None:
            offset = 0
            # `offset` is the position, within the total index tuple,
            # of the first coordinate of a set.
            # This function seems to only make sense if 
            # normalize_index.flatten is True.
            wildcard_sets = {}
            # `wildcard_sets` maps position in the "subsets list"
            # to its set if that set is a wildcard?
            for j,s in enumerate(level.component.index_set().subsets()):
                # Iterate over the sets that could possibly be wildcards
                if s is UnindexedComponent_set:
                    # Why is this treated as a wildcard? (How is this used?)
                    wildcard_sets[j] = s
                    offset += 1
                    continue
                if s.dimen is None:
                    return None
                # `wild` is the number of coordinates of this set (which
                # may be multi-dimensional) that have been sliced.
                wild = sum( 1 for k in range(s.dimen)
                            if k+offset not in level.fixed )
                # `k+offset` is a position in the "total" (flattened)
                # index tuple.
                # All the _slice_generator's information is in terms
                # of this total index tuple.
                if wild == s.dimen:
                    # Every coordinate of set is covered by a wildcard.
                    # This could happen because of explicit slices or
                    # an ellipsis.
                    wildcard_sets[j] = s
                elif wild != 0:
                    # This subset is "touched" by an explicit slice, but
                    # the whole set is not (i.e. there is a fixed
                    # component to this subset).  Therefore, as we
                    # cannot extract that subset, we quit.
                    #
                    # We do not simply `continue` as this "partially sliced
                    # set" has ruined our chance of extracting sets that
                    # perfectly describe our wildcards.
                    return None
                offset += s.dimen
            # I believe that this check is not necessary: weirdnesses
            # with ellipsis should get caught by the check for s.dimen
            # above.
            #
            # Is it not the case that an ellipsis covering a 3-dimen set
            # will result in offset that does not equal
            # explicit_index_count? Or is it that this case is not 
            # exceptional and does not warrant returning None? -RBP
            #
            #if offset != level.explicit_index_count:
            #    return None
            tmp[i] = wildcard_sets
    if not index:
        # index is an empty list, i.e. no wildcard sets
        # have been previously identified.
        # `tmp` will serve as the basis for comparison for future
        # components' wildcard sets.
        return tmp

    # For objects to have "the same" wildcard sets, the same sets must
    # be sliced at the same coordinates of their "subsets list" at the
    # same level of the iter stack.

    # Any of the following would preclude identifying common sets
    # among the objects defined by a slice.
    # However, I can't see a way to actually create any of these
    # situations (i.e., I can't test them).  Assertions are left in for
    # defensive programming.
    assert len(index) == len(tmp)
    # Why must the iter stacks have the same length? If we want
    # `m.b0[:].b1[:].b2.v` and `m.b0[:].b1[:].v` to participate
    # in the same Reference (for some reason), should they not
    # have the same wildcard sets? To support this, only the 
    # number of non-None entries should be compared. (?) -RBP

    for i, level in enumerate(tmp):
        assert (index[i] is None) == (level is None)
        # No slices at this level in the slice
        if level is None:
            continue
        # if there are a differing number of wildcard "subsets"
        if len(index[i]) != len(level):
            return None
        # if any wildcard "subset" differs in position or set.
        if any(index[i].get(j,None) is not _set for j,_set in iteritems(level)):
            return None
        # These checks seem to intentionally preclude
        # `m.b1[:].v` and `m.b2[:,1].v`
        # from having a common set, even if the sliced set is the same.
        # This is probably correct, but additional explanation would
        # be helpful, as this function seems to be doing more than just
        # "identifying wildcard sets."
    return index

def Reference(reference, ctype=_NotSpecified):
    """Creates a component that references other components

    ``Reference`` generates a *reference component*; that is, an indexed
    component that does not contain data, but instead references data
    stored in other components as defined by a component slice.  The
    ctype parameter sets the :py:meth:`Component.type` of the resulting
    indexed component.  If the ctype parameter is not set and all data
    identified by the slice (at construction time) share a common
    :py:meth:`Component.type`, then that type is assumed.  If either the
    ctype parameter is ``None`` or the data has more than one ctype, the
    resulting indexed component will have a ctype of
    :py:class:`IndexedComponent`.

    If the indices associated with wildcards in the component slice all
    refer to the same :py:class:`Set` objects for all data identifed by
    the slice, then the resulting indexed component will be indexed by
    the product of those sets.  However, if all data do not share common
    set objects, or only a subset of indices in a multidimentional set
    appear as wildcards, then the resulting indexed component will be
    indexed by a :py:class:`SetOf` containing a
    :py:class:`_ReferenceSet` for the slice.

    Parameters
    ----------
    reference : :py:class:`IndexedComponent_slice`
        component slice that defines the data to include in the
        Reference component

    ctype : :py:class:`type` [optional]
        the type used to create the resulting indexed component.  If not
        specified, the data's ctype will be used (if all data share a
        common ctype).  If multiple data ctypes are found or type is
        ``None``, then :py:class:`IndexedComponent` will be used.

    Examples
    --------

    .. doctest::

        >>> from pyomo.environ import *
        >>> m = ConcreteModel()
        >>> @m.Block([1,2],[3,4])
        ... def b(b,i,j):
        ...     b.x = Var(bounds=(i,j))
        ...
        >>> m.r1 = Reference(m.b[:,:].x)
        >>> m.r1.pprint()
        r1 : Size=4, Index=r1_index
            Key    : Lower : Value : Upper : Fixed : Stale : Domain
            (1, 3) :     1 :  None :     3 : False :  True :  Reals
            (1, 4) :     1 :  None :     4 : False :  True :  Reals
            (2, 3) :     2 :  None :     3 : False :  True :  Reals
            (2, 4) :     2 :  None :     4 : False :  True :  Reals

    Reference components may also refer to subsets of the original data:

    .. doctest::

        >>> m.r2 = Reference(m.b[:,3].x)
        >>> m.r2.pprint()
        r2 : Size=2, Index=b_index_0
            Key : Lower : Value : Upper : Fixed : Stale : Domain
              1 :     1 :  None :     3 : False :  True :  Reals
              2 :     2 :  None :     3 : False :  True :  Reals

    Reference components may have wildcards at multiple levels of the
    model hierarchy:

    .. doctest::

        >>> from pyomo.environ import *
        >>> m = ConcreteModel()
        >>> @m.Block([1,2])
        ... def b(b,i):
        ...     b.x = Var([3,4], bounds=(i,None))
        ...
        >>> m.r3 = Reference(m.b[:].x[:])
        >>> m.r3.pprint()
        r3 : Size=4, Index=r3_index
            Key    : Lower : Value : Upper : Fixed : Stale : Domain
            (1, 3) :     1 :  None :  None : False :  True :  Reals
            (1, 4) :     1 :  None :  None : False :  True :  Reals
            (2, 3) :     2 :  None :  None : False :  True :  Reals
            (2, 4) :     2 :  None :  None : False :  True :  Reals

    The resulting reference component may be used just like any other
    component.  Changes to the stored data will be reflected in the
    original objects:

    .. doctest::

        >>> m.r3[1,4] = 10
        >>> m.b[1].x.pprint()
        x : Size=2, Index=b[1].x_index
            Key : Lower : Value : Upper : Fixed : Stale : Domain
              3 :     1 :  None :  None : False :  True :  Reals
              4 :     1 :    10 :  None : False : False :  Reals

    """
    if isinstance(reference, IndexedComponent_slice):
        pass
    elif isinstance(reference, Component):
        reference = reference[...]
    else:
        raise TypeError(
            "First argument to Reference constructors must be a "
            "component or component slice (received %s)"
            % (type(reference).__name__,))

    _data = _ReferenceDict(reference)
    _iter = iter(reference)
    if ctype is _NotSpecified:
        ctypes = set()
    else:
        # If the caller specified a ctype, then we will prepopulate the
        # list to improve our chances of avoiding a scan of the entire
        # Reference.
        # This simulates multiple ctypes having been found, so an index
        # of None is sufficient to break the loop.
        ctypes = set((1,2))
    index = []
    for obj in _iter:
        ctypes.add(obj.ctype)
        if not isinstance(obj, ComponentData):
            # This object is not a ComponentData (likely it is a pure
            # IndexedComponent container).  As the Reference will treat
            # it as if it *were* a ComponentData, we will skip ctype
            # identification and return a base IndexedComponent, thereby
            # preventing strange exceptions in the writers and with
            # things like pprint().  Of course, all of this logic is
            # skipped if the User knows better and forced a ctype on us.
            ctypes.add(0)
        if index is not None:
            # Why would index be None here?
            # It got set to None by _identify_wildcard_sets in a previous
            # iteration?
            # ^ This happens if the component _slice_generator in some
            # level of the iter stack is indexed by a set of dimen None.
            #
            # The implication is that index will become a SetOf(_ReferenceSet)?
            index = _identify_wildcard_sets(_iter._iter_stack, index)
        # Note that we want to walk the entire slice, unless we can
        # prove that BOTH there aren't common indexing sets AND there is
        # more than one ctype.
        elif len(ctypes) > 1:
            break
    if not index:
        index = SetOf(_ReferenceSet(reference))
    else:
        wildcards = sum((sorted(iteritems(lvl)) for lvl in index
                         if lvl is not None), [])
        # Wildcards is a list (stack) of (coordinate, set) tuples.
        # Coordinate is that within the subsets list, and set is a
        # wildcard set.
        index = wildcards[0][1]
        # index is the first wildcard set.
        if not isinstance(index, _SetDataBase):
            index = SetOf(index)
        for lvl, idx in wildcards[1:]:
            if not isinstance(idx, _SetDataBase):
                idx = SetOf(idx)
            index = index * idx
        # index is now a SetProduct of the wildcard sets.
    if ctype is _NotSpecified:
        if len(ctypes) == 1:
            ctype = ctypes.pop()
        else:
            ctype = IndexedComponent
    elif ctype is None:
        ctype = IndexedComponent

    obj = ctype(index, ctype=ctype)
    obj._constructed = True
    obj._data = _data
    return obj
