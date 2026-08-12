/*
 * DoubleLinkedList -- classic LRU/FIFO underlying structure with O(1) push/pop
 * both ends, O(1) move-to-front, and stable iteration order.
 *
 * <p>This is intentionally separate from the policy class so policies can be
 * composed (e.g. W-TinyLFU uses one main list + one window FIFO list).
 *
 * <p>Concurrency: all mutators are guarded by the enclosing policy's intrinsic
 * lock. We do not use a concurrent variant because eviction is a second-tier
 * concern (the hot-path touch is the policy's hash, not the linked list).
 */
package org.bytedeco.pytorch.cache.eviction;

import java.util.Iterator;
import java.util.NoSuchElementException;

final class DoubleLinkedList<K> implements Iterable<K> {

    private static final class Node<K> {
        K key;
        Node<K> prev;
        Node<K> next;
        Node(K key) { this.key = key; }
    }

    private final Node<K> head = new Node<>(null); // sentinel
    private final Node<K> tail = new Node<>(null); // sentinel
    private final java.util.HashMap<K, Node<K>> map;
    private int size = 0;

    DoubleLinkedList() {
        map = new java.util.HashMap<>();
        head.next = tail;
        tail.prev = head;
    }

    boolean contains(K key) { return map.containsKey(key); }

    void addFirst(K key) {
        Node<K> n = map.get(key);
        if (n != null) {
            moveToFront(n);
            return;
        }
        n = new Node<>(key);
        map.put(key, n);
        linkFirst(n);
        size++;
    }

    void addLast(K key) {
        Node<K> n = map.get(key);
        if (n != null) {
            unlink(n);
        }
        n = new Node<>(key);
        map.put(key, n);
        linkLast(n);
        size++;
    }

    void moveToFront(K key) {
        Node<K> n = map.get(key);
        if (n == null) return;
        moveToFront(n);
    }

    void moveToBack(K key) {
        Node<K> n = map.get(key);
        if (n == null) return;
        unlink(n);
        linkLast(n);
    }

    void remove(K key) {
        Node<K> n = map.remove(key);
        if (n == null) return;
        unlink(n);
        size--;
    }

    K peekFirst() {
        return head.next == tail ? null : head.next.key;
    }

    K peekLast() {
        return tail.prev == head ? null : tail.prev.key;
    }

    K popFirst() {
        if (head.next == tail) return null;
        Node<K> n = head.next;
        unlink(n);
        map.remove(n.key);
        size--;
        return n.key;
    }

    K popLast() {
        if (tail.prev == head) return null;
        Node<K> n = tail.prev;
        unlink(n);
        map.remove(n.key);
        size--;
        return n.key;
    }

    int size() { return size; }

    boolean isEmpty() { return size == 0; }

    void clear() {
        head.next = tail;
        tail.prev = head;
        map.clear();
        size = 0;
    }

    @Override
    public Iterator<K> iterator() {
        return new Iterator<K>() {
            Node<K> cursor = head.next;
            @Override public boolean hasNext() { return cursor != tail; }
            @Override public K next() {
                if (cursor == tail) throw new NoSuchElementException();
                K k = cursor.key;
                cursor = cursor.next;
                return k;
            }
        };
    }

    private void linkFirst(Node<K> n) {
        n.prev = head;
        n.next = head.next;
        head.next.prev = n;
        head.next = n;
    }

    private void linkLast(Node<K> n) {
        n.prev = tail.prev;
        n.next = tail;
        tail.prev.next = n;
        tail.prev = n;
    }

    private void unlink(Node<K> n) {
        n.prev.next = n.next;
        n.next.prev = n.prev;
        n.prev = null;
        n.next = null;
    }

    private void moveToFront(Node<K> n) {
        unlink(n);
        linkFirst(n);
    }
}
