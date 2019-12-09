#include <stdlib.h>
#include <string.h>
#include "list.h"
#include "option_list.h"

// list is a linked list implementtion


/**
 * Creates an empty list.
 * This list is allocated by make_list but must be freed by the caller.
 * @return pointer to new list
 */
list* make_list()
{
    list* l = (list*)malloc(sizeof(list));
    l->size = 0;
    l->front = NULL;
    l->back = NULL;
    return l;
}


/**
 * Pop last item from a list.
 * The node struct containing the item is freed.
 * The list is modified in-place.
 * @param l pointer to the list
 * @return pointer to value that was popped (or NULL if the list was empty)
 */
void* list_pop(list* const l){
    if (!l->back) {
        // list is empty
        return NULL;
    }

    node* b = l->back;
    void* val = b->val;

    l->back = b->prev;

    if (l->back != NULL) {
        // remove the previous node's reference to the popped node
        l->back->next = NULL;
    }

    // free the popped node
    free(b);

    // decrement the list's size field
    l->size--;

    return val;
}


/**
 * Appends a value to a list.
 * This modifies the list in-place.
 * @param l   pointer to an existing list.
 * @param val pointer to value to be appended.
 */
void list_insert(list* const l, void* const val)
{
    // create new node
    node* newnode = (node*)malloc(sizeof(node));
    newnode->val = val;
    newnode->next = 0;

    if (!l->back) {
        // list is empty
        l->front = newnode;
        newnode->prev = NULL;

    } else {
        // list is non-empty
        l->back->next = newnode;
        newnode->prev = l->back;
    }

    l->back = newnode;
    l->size++;
}


/**
 * Free a node as well as every node pointed it points to, recursively.
 * The values stored in the nodes are not freed.
 * @param n pointer to node
 */
void free_node(node* n)
{
    node* next;
    while (n != NULL) {
        next = n->next;
        free(n);
        n = next;
    }
}


void free_list_val(list *l)
{
    node *n = l->front;
    node *next;
    while (n) {
        next = n->next;
        free(n->val);
        n = next;
    }
}


/**
 * Free the memory allocated for a list and all its nodes.
 * The values stored in the list are not freed.
 * @param l pointer to list to free
 */
void free_list(list* const l)
{
    free_node(l->front);
    free(l);
}


/**
 * Call free() on every value stored in a list.
 * @param l pointer to the list
 */
void free_list_contents(const list* const l)
{
    node* n = l->front;
    while (n != NULL) {
        free(n->val);
        n = n->next;
    }
}


/**
 * Given a list of kvp (key-value pair) structs, free the memory allocated for
 *     the kvp structs and for the keys.
 * The kvp structs' values are not freed.
 * The list is not modified (so it now contains pointers to undefined data).
 * @param l pointer to a list contaning kvp structs
 */
void free_list_contents_kvp(const list* const l)
{
    node* n = l->front;
    while (n != NULL) {
        kvp* p = n->val;
        free(p->key);
        free(p);
        n = n->next;
    }
}


/**
 * Convert a list to an array.
 * All the values are taken from the list and put into a C array of the right
 *     size that is newly allocated.
 * The caller must free the new array.
 * The list is not modified.
 * @param l pointer to the list to convert
 */
void** list_to_array(const list* const l)
{
    void** a = calloc(l->size, sizeof(void*));
    int count = 0;
    node* n = l->front;
    while (n != NULL) {
        a[count] = n->val;

        count++;
        n = n->next;
    }
    return a;
}
