#ifndef LIST_H
#define LIST_H


typedef struct node{
    void* val;
    struct node* next;
    struct node* prev;
} node;


typedef struct list{
    int size;
    node* front;
    node* back;
} list;


#ifdef __cplusplus
extern "C" {
#endif

list* make_list();
int list_find(list* l, void* val);

void list_insert(list* const l, void* const val);

void** list_to_array(const list* const l);

void free_list(list* const l);
void free_list_contents(const list* const l);
void free_list_contents_kvp(const list* const l);

#ifdef __cplusplus
}
#endif
#endif
