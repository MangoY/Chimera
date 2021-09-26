from pycparser import parse_file, c_generator
from pycparser import c_parser, c_ast
#from pycparser.c_ast import NodeVisitor
import sys

sys.path.extend(['.', '..'])

class NodeVisitor(object):
    def visit(self, node, scope=None):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, scope)

    def generic_visit(self, node, scope):
        """Called if no explicit visitor function exists for a node."""
        for field, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value, scope)

class NodeTransformer(NodeVisitor):
    def generic_visit(self, node, scope=None):
        if(isinstance(node, c_ast.Node)):
            #print('NODE:'+node.__class__.__name__)
            for field in dir(node):
                old_value = getattr(node, field)
                if isinstance(old_value, list):
                    new_value = []
                    for value in old_value:
                        value = self.visit(value, scope)
                        new_value.append(value)
                    setattr(node, field, new_value)
                elif isinstance(old_value, c_ast.Node):
                    new_node = self.visit(old_value, scope)
                    if new_node is None:
                        delattr(node, field)
                    else:
                        setattr(node, field, new_node)
                else:
                    pass # do nothing if it's not expected types
        return node

class MyTransformer(NodeTransformer):
    def __init__(self):
        self.scopes = {}
        self.tunable_params = []
        
    def visit_For(self, node, scope):
        """For loop"""
        # routine, transform all the child first
        for field in dir(node):
            old_value = getattr(node, field)
            if isinstance(old_value, list):
                new_value = []
                for value in old_value:
                    value = self.visit(value, scope)
                    new_value.append(value)
            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value, scope)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
            else:
                pass # do nothing if it's not expected types
        
        init_stmt = node.init
        cond_stmt = node.cond
        next_stmt = node.next
        body_stmt = node.stmt
        
        # define loop interchange function for moving the thread loops in
        def loop_interchange(root_node, scope):
            loop_identifier = scope+'_{0}_Z'.format(self.scopes[scope])
            root_node = c_ast.Label(loop_identifier, root_node)
            
            # record tunable parameter
            # TODO: wrap this in a function
            self.tunable_params.append(','.join(['loop',scope,loop_identifier,'BLOCKDIM_Z']))
            
            node = root_node.stmt
            # skip through the thread loops, check if the structure is as expected
            # also add labels to all thread loop layers
            for i in range(3):
                if(isinstance(node, c_ast.For)):
                    try:
                        loop_idx_struct_name = node.init.lvalue.name.name
                    except AttributeEerror:
                        print('cannot get loop index thread loops!\n')
                        return root_node
                    if(loop_idx_struct_name != 'get_local_id'):
                        print('not local id thread loops!\n')
                        return root_node
                    thread_loop_inner = node
                    if (i != 2): # do not add label for the inner most layer
                        # first determine a unique lable for the loop
                        
                        if (i == 0):
                            loop_identifier = scope+'_{0}_Y'.format(self.scopes[scope])
                            self.tunable_params.append(','.join(['loop',scope,loop_identifier,'BLOCKDIM_Y']))
                        else:
                            loop_identifier = scope+'_{0}_X'.format(self.scopes[scope])
                            self.tunable_params.append(','.join(['loop',scope,loop_identifier,'BLOCKDIM_X']))
                        node.stmt.block_items[0] = c_ast.Label(loop_identifier, node.stmt.block_items[0])
                        node = node.stmt.block_items[0].stmt
                    else:
                        node = node.stmt.block_items[0] # TODO: potential bug, hardcoding block_items[1]
                else:
                    return root_node

            # the thread loop trio is identified and verified at this point
            self.scopes[scope] = self.scopes[scope] + 1
            
            # dig to the inner most perfect loop layer
            parent_node = node # this will ensure that if stmt after the thread loops is not for loop 
                               #(or wrapped with Compound) the final check for transform-ability will fail
            while(1):
                if(isinstance(node, c_ast.For)):
                    try:
                        loop_idx_struct_name = node.init.lvalue.name.name
                    except AttributeError:
                        loop_idx_struct_name = None
                    if(loop_idx_struct_name == 'get_local_id'):
                        print('not interchange-able innerloop\n')
                        break
                    else:
                        parent_node = node
                        node = node.stmt
                elif isinstance(node, c_ast.Compound):
                    sub_loops = 0
                    loops = []
                    for item in node.block_items:
                        if isinstance(item, c_ast.Pragma):
                            continue
                        elif isinstance(item, c_ast.For):
                            sub_loops = sub_loops + 1
                            loops.append(item)
                        else:
                            break
                    if(sub_loops == 1):
                        parent_node = node
                        node = loops[0]
                    else:
                        break
                else:
                    break

            # check transform-ability
            if (not isinstance(parent_node, c_ast.For)):
                return root_node
            else:
                thread_loop_inner.stmt = parent_node.stmt
                parent_node.stmt = root_node
                return parent_node
           
            return root_node # default case
        # loop interchange function end
        
        # start replacing loop boundary and perform loop interchange
        if (isinstance(init_stmt, c_ast.Assignment)):
            if(isinstance(init_stmt.lvalue, c_ast.StructRef)):
                loop_idx_struct_name = node.init.lvalue.name.name
                loop_idx_field_name = node.init.lvalue.field.name
                
                if(loop_idx_struct_name == 'get_local_id'):
                    if(loop_idx_field_name == 'x'):
                        cond_stmt.right = c_ast.ID(name='BLOCKDIM_X')
                    elif(loop_idx_field_name == 'y'):
                        cond_stmt.right = c_ast.ID(name='BLOCKDIM_Y')
                    elif(loop_idx_field_name == 'z'):
                        cond_stmt.right = c_ast.ID(name='BLOCKDIM_Z')
                
                if(loop_idx_struct_name == 'get_local_id' and loop_idx_field_name == 'z'):
                    new_node = loop_interchange(node, scope)
                    return new_node
        return node
    
    def visit_FuncDef(self, node, scope):
        # for function definition we need to first extract the name
        # have to do it here since the name come from the decl but need to be
        # passed to the body
        if(isinstance(node.decl.type, c_ast.FuncDecl)):
            func_name = node.decl.name
            self.scopes[func_name] = 0
    
        # routine, transform all the child first
        for field in dir(node):
            # only pass the scope to body
            # to avoid some problems
            if (field=='body'):
                scope = func_name
            else:
                scope = None
            old_value = getattr(node, field)
            if isinstance(old_value, list):
                new_value = []
                for value in old_value:
                    value = self.visit(value, scope)
                    new_value.append(value)
            elif isinstance(old_value, c_ast.Node):
                new_node = self.visit(old_value, scope)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
            else:
                pass # do nothing if it's not expected type
        return node
    
    def visit_ArrayDecl(self, node, scope):
        """array decleration, need to add array partitioning pragma after this"""
        
        def find_dims(array):
            dim = 0
            if (isinstance(array.dim, c_ast.Constant)):
                dim = array.dim.value
            elif (isinstance(array.dim, c_ast.ID)):
                dim = array.dim.name
            else:
                raise AttributeError('Invalid array dim type ')
                
            if(not isinstance(array.type, c_ast.ArrayDecl)):
                array_name = array.type.declname
                return [dim], array_name
            else:
                dims, array_name = find_dims(array.type)
                return dims+[dim], array_name
        
        # only perform analysis for loops in the function body
        if(scope != None):
            # acquire dimensions and array name    
            dims, array_name = find_dims(node)
            for i, dim in enumerate(dims):
                self.tunable_params.append(','.join(['array', scope, array_name, str(dim), str(i)]))
        return node
    
    def get_tunable_params(self):
        return self.tunable_params


def transform_and_extract(source_path, transformed_source_path, params_path):
    trv = MyTransformer()
    ast = parse_file(source_path, use_cpp=True)
    trv.visit(ast)
    generator = c_generator.CGenerator()

    with open(transformed_source_path, 'w') as file:
        file.write(generator.visit(ast))

    with open(params_path, 'w') as file:
        file.write(','.join(['type', 'scope', 'name', 'range', 'dim']) + '\n')
        file.write('\n'.join(trv.get_tunable_params()))


if __name__ == "__main__":
    transform_and_extract('./test_files/test.cpp',
                          './test_files/transformed_source.cpp',
                          './test_files/tunable_params.csv')


