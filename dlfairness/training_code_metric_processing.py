import ast
import asttokens
import re
import argparse
import astunparse
import json
import copy
from collections import deque


def extract_metric(filepath):
    with open(filepath, "r") as source:
        atok = asttokens.ASTTokens(source.read(), parse=True)
        # tree = ast.parse(source.read())

    funcDefDic = createFunctDefMap(atok)

    #   for func_obj in funcDefNodes.keys():
    #       print (str(func_obj.target) + ", size: %s" % len(funcDefNodes[func_obj]))
    #       for funcdef in funcDefNodes[func_obj]:
    #           print ("\t%s" % funcdef.name)

    loopAnalyzer = MainLoopsSearcher(atok, funcDefDic)
    loopAnalyzer.visit(atok.tree)
    return loopAnalyzer.main_loops


def extract_metric_without_main_loop(filepath):
    with open(filepath, "r") as source:
        atok = asttokens.ASTTokens(source.read(), parse=True)
        # tree = ast.parse(source.read())

    metricAnalyzer = LoggingSearcher(atok)
    metricAnalyzer.visit(atok.tree)
    return metricAnalyzer.funcs


def createFunctDefMap(atok):
    funcdefVisitor = FuncDefVisitor(atok)
    funcdefVisitor.visit(atok.tree)
    funcdefs_dict = funcdefVisitor.func_defs  # key: func_name, value: func_def_node

    loopVisitor = LoopVisitorForFuncDefs(atok, **funcdefs_dict)
    loopVisitor.visit(atok.tree)
    return loopVisitor.call_graph_map  # key: loop node, value: [defs node of called functions in the loop]


class FuncDefVisitor(ast.NodeVisitor):
    def __init__(self, atok):
        self.atok = atok
        self.func_defs = {}

    def visit_FunctionDef(self, node):
        func_name = node.__dict__['name']
        self.func_defs[func_name] = node


class LoopVisitorForFuncDefs(ast.NodeVisitor):
    def __init__(self, atok, **funcdefs_dict):
        self.atok = atok
        self.funcdefs_dict = funcdefs_dict
        self.call_graph_map = {}

    def visit_For(self, node):
        self.call_graph_map[node] = self.get_func_calls_in_loop(node)

    def visit_While(self, node):
        self.call_graph_map[node] = self.get_func_calls_in_loop(node)

    def get_func_calls_in_loop(self, node):
        funccallVisitor = FuncDefForFuncCall(self.atok, **self.funcdefs_dict)
        funccallVisitor.visit(node)
        return funccallVisitor.called_func_defs


class FuncDefForFuncCall(ast.NodeVisitor):
    def __init__(self, atok, **funcdefs_dict):
        self.atok = atok
        self.funcdefs_dict = funcdefs_dict
        self.called_func_defs = []

    def visit_Call(self, node):
        func_node = node.func
        if func_node.__class__.__name__ == 'Name':
            funcn = func_node.id
        elif func_node.__class__.__name__ == 'Attribute':
            funcn = func_node.attr
        else:
            funcn = None

        if funcn in self.funcdefs_dict.keys():
            if funcn is not None:
                # print (funcn)
                self.called_func_defs.append(self.funcdefs_dict[funcn])
            else:
                print("ERROR! None should not be a function def: %s" % self.funcdefs_dict[funcn].name)


class MainLoopsSearcher(ast.NodeVisitor):
    def __init__(self, atok, funcDefDic):
        self.atok = atok
        self.funcDefDic = funcDefDic
        self.main_loops = []

    def visit_For(self, node):
        funcs, NI_metrics = self.get_metrics(node)
        target = node.target
        if target.__class__.__name__ == 'Tuple':
            loop_ii = self.atok.get_text(target.elts[0])
        else:
            loop_ii = self.atok.get_text(target)
        if loop_ii in NI_metrics:
            self.main_loops.append((node, loop_ii, target.lineno, funcs, NI_metrics))

    def visit_While(self, node):
        funcs, NI_metrics = self.get_metrics(node)
        for n in ast.walk(node):
            for child in ast.iter_child_nodes(n):
                var = self.atok.get_text(child)
                for NI_metrics in NI_metrics:
                    if var in NI_metrics:
                        loop_ii = self.atok.get_text(child)
                        self.main_loops.append((node, loop_ii, node.lineno, funcs, NI_metrics))
                        return

    def get_metrics(self, node):
        metricAnalyzer = LoggingSearcher(self.atok)
        metricAnalyzer.visit(node)

        funcs = metricAnalyzer.funcs
        NI_metrics = metricAnalyzer.NI_metrics

        funcDefs = self.funcDefDic[node]
        for funcDef in funcDefs:
            metricAnalyzer = LoggingSearcher(self.atok)
            metricAnalyzer.visit(funcDef)

            funcs.extend(metricAnalyzer.funcs)
            NI_metrics.extend(metricAnalyzer.NI_metrics)
        return funcs, NI_metrics


class LoggingSearcher(ast.NodeVisitor):
    def __init__(self, atok):
        self.atok = atok
        self.funcs = []
        self.NI_metrics = []

    def visit_Call(self, node):
        func_node = node.func
        if func_node.__class__.__name__ == 'Name':
            funcn = func_node.id
        elif func_node.__class__.__name__ == 'Attribute':
            funcn = func_node.attr
        else:
            funcn = None

        if funcn in ['write', 'print', 'log']:
            print_interal_visitor = MetricsSearcher(self.atok)
            print_interal_visitor.visit(node)

            self.funcs.append((func_node, funcn, func_node.lineno, print_interal_visitor.metrics))
            self.NI_metrics.extend(print_interal_visitor.NI_metrics)
        elif funcn in ['info']:
            if len(node.args) > 1:
                info_metrics = []
                info_NI_metrics = []

                format_str = node.args[0].s
                holders_idx = extract_mod(format_str)

                for expr_idx, expr in enumerate(node.args):
                    if expr_idx > 0:
                        if (expr_idx - 1) in holders_idx:
                            info_metrics.append((copy.deepcopy(expr), self.atok.get_text(expr), expr.lineno))
                        else:
                            info_NI_metrics.append(self.atok.get_text(expr))
            else:
                print_interal_visitor = MetricsSearcher(self.atok)
                print_interal_visitor.visit(node)

                info_metrics = print_interal_visitor.metrics
                info_NI_metrics = print_interal_visitor.NI_metrics

            self.funcs.append((func_node, funcn, func_node.lineno, info_metrics))
            self.NI_metrics.extend(info_NI_metrics)
        else:
            self.generic_visit(node)


class MetricsSearcher(ast.NodeVisitor):
    def __init__(self, atok):
        self.atok = atok
        self.metrics = []
        self.NI_metrics = []

    def visit_Call(self, node):
        func_node = node.func
        if func_node.__class__.__name__ == 'Attribute':
            funcn = func_node.attr
            funcv = func_node.value
        else:
            funcn = None

        if funcn == 'format':
            if funcv.__class__.__name__ == 'Name':
                # Todo: Get the variable and analyse that variable
                format_str = ''
            elif funcv.__class__.__name__ == 'Str':
                format_str = funcv.s
            else:
                format_str = ''

            accesses_dic, NI_accesses_dic = extract_format(format_str)

            for exprIdx, expr in enumerate(node.args):
                if str(exprIdx) in accesses_dic:
                    self.metrics.append((copy.deepcopy(expr), self.atok.get_text(expr), expr.lineno))
                else:
                    self.NI_metrics.append(self.atok.get_text(expr))

            for keyword in node.keywords:
                arg = keyword.arg
                expr = keyword.value

                if arg in accesses_dic:
                    access = accesses_dic[arg]
                    if access == None:
                        self.metrics.append((copy.deepcopy(expr), self.atok.get_text(expr), expr.lineno))
                    else:
                        var = ast.Attribute(value=copy.deepcopy(expr), attr=access, ctx=ast.Load())
                        self.metrics.append((var, self.atok.get_text(expr) + '.' + access, expr.lineno))

                if arg in NI_accesses_dic:
                    access = NI_accesses_dic[arg]
                    if access == None:
                        self.NI_metrics.append(self.atok.get_text(expr))
                    else:
                        self.NI_metrics.append(self.atok.get_text(expr) + '.' + access)

        else:
            self.generic_visit(node)

    def visit_BinOp(self, node):
        if node.left.__class__.__name__ == 'Str' and node.op.__class__.__name__ == 'Mod':
            format_str = node.left.s
            holders_idx = extract_mod(format_str)
            if node.right.__class__.__name__ == 'Tuple':
                for expr_idx, expr in enumerate(node.right.elts):
                    if expr_idx in holders_idx:
                        self.metrics.append((copy.deepcopy(expr), self.atok.get_text(expr), expr.lineno))
                    else:
                        self.NI_metrics.append(self.atok.get_text(expr))
            else:
                if 0 in holders_idx:
                    self.metrics.append((copy.deepcopy(node.right), self.atok.get_text(node.right), node.right.lineno))
                else:
                    self.NI_metrics.append(self.atok.get_text(node.right))
        else:
            self.generic_visit(node)


def extract_format(str_with_format):
    accesses = re.findall('\{(.*?)\}', str_with_format)
    accesses_dic = {}
    NI_accesses_dic = {}
    accessIdx = 0
    for access in accesses:
        splitted_access = access.split(':')

        var = splitted_access[0]
        if var == '':
            var = str(accessIdx)
            accessIdx = accessIdx + 1

        if len(splitted_access) > 1:
            formatting_str = splitted_access[1]
        else:
            formatting_str = ''

        splitted_var = var.split('.', 1)
        arg = splitted_var[0]
        if len(splitted_var) > 1:
            expr = splitted_var[1]
        else:
            expr = None

        if valid_format(formatting_str):
            accesses_dic[arg] = expr
        else:
            NI_accesses_dic[arg] = expr

    return accesses_dic, NI_accesses_dic


def extract_mod(str_with_format):
    holders = re.findall('\%(?:.*?)[sdf]', str_with_format)
    holders_idx = []
    for accessIdx, holder in enumerate(holders):
        if valid_format(holder):
            holders_idx.append(accessIdx)
    return holders_idx


def valid_format(formatting_str):
    return '.' in formatting_str or 'f' in formatting_str


def modify_file(filepath, loops, outfilepath, logfilepath):
    with open(filepath, "r") as source:
        atok = asttokens.ASTTokens(source.read(), parse=True)
        # tree = ast.parse(source.read())

    funcDefDic = createFunctDefMap(atok)

    analyzer = MainLoopsSearcher(atok, funcDefDic)
    analyzer.visit(atok.tree)
    loops = analyzer.main_loops

    # Add backward link to parent
    for node in ast.walk(atok.tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    # add import
    index = 0
    last_from_future_index = 0
    for top_statement in atok.tree.body:
        if top_statement.__class__.__name__ == 'ImportFrom' and top_statement.module == '__future__':
            last_from_future_index = index
        index = index + 1

    import_statement = ast.ImportFrom(module='dl_logging_helper', names=[ast.alias(name='DLVarLogger', asname=None)], level=0)
    atok.tree.body.insert(last_from_future_index + 1, import_statement)

    modifier = MainLoopsModifier(loops)
    modifier.visit(atok.tree)

    source = astunparse.unparse(atok.tree)
    with open(outfilepath, "w") as out_file:
        out_file.write(source)

    with open(logfilepath, "w") as out_file:
        out_file.write('Iter,Time')
        for metric in modifier.metrics:
            out_file.write(',%s' % metric)
        out_file.write('\n')


def modify_file_with_metrics(filepath, funcs, outfilepath, logfilepath):
    with open(filepath, "r") as source:
        atok = asttokens.ASTTokens(source.read(), parse=True)
        # tree = ast.parse(source.read())

    metricAnalyzer = LoggingSearcher(atok)
    metricAnalyzer.visit(atok.tree)
    funcs = metricAnalyzer.funcs

    # Add backward link to parent
    for node in ast.walk(atok.tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    # add import
    index = 0
    last_from_future_index = 0
    for top_statement in atok.tree.body:
        if top_statement.__class__.__name__ == 'ImportFrom' and top_statement.module == '__future__':
            last_from_future_index = index
        index = index + 1

    import_statement = ast.ImportFrom(module='dl_logging_helper_no_loop', names=[ast.alias(name='DLVarLogger', asname=None)], level=0)
    atok.tree.body.insert(last_from_future_index + 1, import_statement)

    modifier = LoggingModifier(funcs)
    modifier.visit(atok.tree)

    source = astunparse.unparse(atok.tree)
    with open(outfilepath, "w") as out_file:
        out_file.write(source)

    with open(logfilepath, "w") as out_file:
        out_file.write('Iter,Time')
        for metric in modifier.metrics:
            out_file.write(',%s' % metric)
        out_file.write('\n')


class MainLoopsModifier(ast.NodeVisitor):
    def __init__(self, loops):
        self.main_loops = loops
        self.metrics = []

    def visit_For(self, node):
        self.process_loop(node)

    def visit_While(self, node):
        self.process_loop(node)

    def process_loop(self, node):
        for loop in self.main_loops:
            (loop_node, loop_var, loop_line, funcs, NI_metrics) = loop
            if node.lineno == loop_line:
                # add begin loop
                begin_statement = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='DLVarLogger'), attr='beginLoop'), args=[ast.Name(id=loop_var)], keywords=[]))
                node.body.insert(0, begin_statement)
                # add end loop
                end_statement = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='DLVarLogger'), attr='endLoop'), args=[], keywords=[]))
                node.body.append(end_statement)

                # add logging for metrics
                modifier = LoggingModifier(funcs)
                modifier.visit(node)

                self.metrics.extend(modifier.metrics)

                end_log_statement = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='DLVarLogger'), attr='endLogger'), args=[], keywords=[]))
                node.parent.body.append(end_log_statement)

class LoggingModifier(ast.NodeVisitor):
    def __init__(self, funcs):
        self.funcs = funcs
        self.metrics = []

    def visit_Call(self, node):
        for func in self.funcs:
            (func_node, funcn, func_line, metrics) = func
            if hasattr(node, 'lineno') and node.lineno == func_line:
                statement_list = node.parent.parent.body
                if node.parent in statement_list:
                    func_index = statement_list.index(node.parent)
                    for metric in metrics:
                        (metric_node, metric_str, metric_line) = metric
                        print('Adding code to log ' + metric_str)
                        log_func = ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='DLVarLogger'), attr='log'), args=[ast.Str(s=metric_str), metric_node], keywords=[]))
                        statement_list.insert(func_index + 1, log_func)
                        self.metrics.append(metric_str)
                return
