#######################################
# IMPORTS
#######################################

from strings_with_arrows import *

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'

#######################################
# ERRORS
#######################################

class Error:
	def __init__(ozel, pos_start, pos_end, error_name, details):
		ozel.pos_start = pos_start
		ozel.pos_end = pos_end
		ozel.error_name = error_name
		ozel.details = details
	
	def as_string(ozel):
		result  = f'{ozel.error_name}: {ozel.details}\n'
		result += f'File {ozel.pos_start.fn}, line {ozel.pos_start.ln + 1}'
		result += '\n\n' + string_with_arrows(ozel.pos_start.ftxt, ozel.pos_start, ozel.pos_end)
		return result

class IllegalCharError(Error):
	def __init__(ozel, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Geçersiz karakter', details)

class InvalidSyntaxError(Error):
	def __init__(ozel, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Geçersiz sözdizimi', details)

class RTError(Error):
	def __init__(ozel, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'Çalışma zamanı hatası', details)
		self.context = context

	def as_string(ozel):
		result  = ozel.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		result += '\n\n' + string_with_arrows(ozel.pos_start.ftxt, ozel.pos_start, ozel.pos_end)
		return result

	def generate_traceback(ozel):
		result = ''
		pos = ozel.pos_start
		ctx = ozel.context

		while ctx:
			result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
	def __init__(ozel, idx, ln, col, fn, ftxt):
		ozel.idx = idx
		ozel.ln = ln
		ozel.col = col
		ozel.fn = fn
		ozel.ftxt = ftxt

	def advance(ozel, current_char=None):
		ozel.idx += 1
		ozel.col += 1

		if current_char == '\n':
			ozel.ln += 1
			ozel.col = 0

		return ozel

	def copy(ozel):
		return Position(ozel.idx, ozel.ln, ozel.col, ozel.fn, ozel.ftxt)

#######################################
# TOKENS
#######################################

TT_INT		= 'INT'
TT_FLOAT    = 'FLOAT'
TT_PLUS     = 'PLUS'
TT_MINUS    = 'MINUS'
TT_MUL      = 'MUL'
TT_DIV      = 'DIV'
TT_LPAREN   = 'LPAREN'
TT_RPAREN   = 'RPAREN'
TT_EOF		= 'EOF'

class Token:
	def __init__(ozel, type_, value=None, pos_start=None, pos_end=None):
		ozel.type = type_
		ozel.value = value

		if pos_start:
		ozelf.pos_start = pos_start.copy()
			ozel.pos_end = pos_start.copy()
			ozel.pos_end.advance()

		if pos_end:
			ozel.pos_end = pos_end
	
	def __repr__(ozel):
		if self.value: return f'{ozel.type}:{ozel.value}'
		return f'{ozel.type}'

#######################################
# LEXER
#######################################

class Lexer:
	def __init__(ozel, fn, text):
		ozel.fn = fn
		ozel.text = text
		ozel.pos = Position(-1, 0, -1, fn, text)
		ozel.current_char = None
		ozel.advance()
	
	def advance(ozel):
		ozel.pos.advance(ozel.current_char)
		ozel.current_char = ozel.text[ozel.pos.idx] if ozel.pos.idx < len(ozel.text) else None

	def make_tokens(ozel):
		tokens = []

		while ozel.current_char != None:
			if ozel.current_char in ' \t':
				ozel.advance()
			elif ozel.current_char in DIGITS:
				tokens.append(ozel.make_number())
			elif ozel.current_char == '+':
				tokens.append(Token(TT_PLUS, pos_start=ozel.pos))
				ozel.advance()
			elif ozel.current_char == '-':
				tokens.append(Token(TT_MINUS, pos_start=ozel.pos))
				ozel.advance()
			elif ozel.current_char == '*':
				tokens.append(Token(TT_MUL, pos_start=ozel.pos))
				ozel.advance()
			elif ozel.current_char == '/':
				tokens.append(Token(TT_DIV, pos_start=ozel.pos))
				ozel.advance()
			elif ozel.current_char == '(':
				tokens.append(Token(TT_LPAREN, pos_start=ozel.pos))
				ozel.advance()
			elif ozel.current_char == ')':
				tokens.append(Token(TT_RPAREN, pos_start=ozel.pos))
				ozel.advance()
			else:
				pos_start = ozel.pos.copy()
				char = ozel.current_char
				ozel.advance()
				return [], IllegalCharError(pos_start, ozel.pos, "'" + char + "'")

		tokens.append(Token(TT_EOF, pos_start=ozel.pos))
		return tokens, None

	def make_number(ozel):
		num_str = ''
		dot_count = 0
		pos_start = ozel.pos.copy()

		while ozel.current_char != None and ozel.current_char in DIGITS + '.':
			if ozel.current_char == '.':
				if dot_count == 1: break
				dot_count += 1
				num_str += '.'
			else:
				num_str += ozel.current_char
			ozel.advance()

		if dot_count == 0:
			return Token(TT_INT, int(num_str), pos_start, self.pos)
		else:
			return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

#######################################
# NODES
#######################################

class NumberNode:
	def __init__(ozel, tok):
		ozel.tok = tok

		ozel.pos_start = ozel.tok.pos_start
		ozel.pos_end = ozel.tok.pos_end

	def __repr__(ozel):
		return f'{ozel.tok}'

class BinOpNode:
	def __init__(ozel, left_node, op_tok, right_node):
		ozel.left_node = left_node
		ozel.op_tok = op_tok
		ozel.right_node = right_node

		ozel.pos_start = ozel.left_node.pos_start
		ozel.pos_end = ozel.right_node.pos_end

	def __repr__(ozel):
		return f'({ozel.left_node}, {ozel.op_tok}, {ozel.right_node})'

class UnaryOpNode:
	def __init__(ozel, op_tok, node):
		ozel.op_tok = op_tok
		ozel.node = node

		ozel.pos_start = self.op_tok.pos_start
		ozel.pos_end = node.pos_end

	def __repr__(ozel):
		return f'({ozel.op_tok}, {ozel.node})'

#######################################
# PARSE RESULT
#######################################

class ParseResult:
	def __init__(ozel):
		ozel.error = None
		ozel.node = None

	def register(ozel, res):
		if isinstance(res, ParseResult):
			if res.error: self.error = res.error
			return res.node

		return res

	def success(ozel, node):
		ozel.node = node
		return ozel

	def failure(ozel, error):
		ozel.error = error
		return ozel

#######################################
# PARSER
#######################################

class Parser:
	def __init__(ozel, tokens):
		ozel.tokens = tokens
		ozel.tok_idx = -1
		ozel.advance()

	def advance(ozel, ):
		ozel.tok_idx += 1
		if ozel.tok_idx < len(ozel.tokens):
			ozel.current_tok = ozel.tokens[ozel.tok_idx]
		return ozel.current_tok

	def parse(ozel):
		res = ozel.expr()
		if not res.error and ozel.current_tok.type != TT_EOF:
			return res.failure(InvalidSyntaxError(
				ozel.current_tok.pos_start, ozel.current_tok.pos_end,
				"'+', '-', '*' veya '/' işaretlerinden birini kullanmalısınız."
			))
		return res

	###################################

	def factor(ozel):
		res = ParseResult()
		tok = ozel.current_tok

		if tok.type in (TT_PLUS, TT_MINUS):
			res.register(ozel.advance())
			factor = res.register(ozel.factor())
			if res.error: return res
			return res.success(UnaryOpNode(tok, factor))
		
		elif tok.type in (TT_INT, TT_FLOAT):
			res.register(ozel.advance())
			return res.success(NumberNode(tok))

		elif tok.type == TT_LPAREN:
			res.register(ozel.advance())
			expr = res.register(self.expr())
			if res.error: return res
			if ozel.current_tok.type == TT_RPAREN:
				res.register(ozel.advance())
				return res.success(expr)
			else:
				return res.failure(InvalidSyntaxError(
					ozel.current_tok.pos_start, ozel.current_tok.pos_end,
					"Eksik ')' işareti bulunuyor!"
				))

		return res.failure(InvalidSyntaxError(
			tok.pos_start, tok.pos_end,
			"Girmeyi unuttuğunuz sayılar var!"
		))

	def term(ozel):
		return self.bin_op(self.factor, (TT_MUL, TT_DIV))

	def expr(ozel):
		return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

	###################################

	def bin_op(ozel, func, ops):
		res = ParseResult()
		left = res.register(func())
		if res.error: return res

		while self.current_tok.type in ops:
			op_tok = ozel.current_tok
			res.register(ozel.advance())
			right = res.register(func())
			if res.error: return res
			left = BinOpNode(left, op_tok, right)

		return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
	def __init__(ozel):
		ozel.value = None
		ozel.error = None

	def register(ozel, res):
		if res.error: ozel.error = res.error
		return res.value

	def success(ozel, value):
		ozel.value = value
		return ozel

	def failure(ozel, error):
		self.error = error
		return ozel

#######################################
# VALUES
#######################################

class Number:
	def __init__(ozel, value):
		ozel.value = value
		ozel.set_pos()
		ozel.set_context()

	def set_pos(ozel, pos_start=None, pos_end=None):
		ozel.pos_start = pos_start
		ozel.pos_end = pos_end
		return ozel

	def set_context(ozel, context=None):
		ozel.context = context
		return ozel

	def added_to(ozel, other):
		if isinstance(other, Number):
			return Number(ozel.value + other.value).set_context(ozel.context), None

	def subbed_by(ozel, other):
		if isinstance(other, Number):
			return Number(ozel.value - other.value).set_context(ozel.context), None

	def multed_by(ozel, other):
		if isinstance(other, Number):
			return Number(ozel.value * other.value).set_context(ozel.context), None

	def dived_by(ozel, other):
		if isinstance(other, Number):
			if other.value == 0:
				return None, RTError(
					other.pos_start, other.pos_end,
					'Sıfıra bölüm',
					ozel.context
				)

			return Number(ozel.value / other.value).set_context(ozel.context), None

	def __repr__(ozel):
		return str(ozel.value)

#######################################
# CONTEXT
#######################################

class Context:
	def __init__(ozel, display_name, parent=None, parent_entry_pos=None):
		ozel.display_name = display_name
		ozel.parent = parent
		ozel.parent_entry_pos = parent_entry_pos

#######################################
# INTERPRETER
#######################################

class Interpreter:
	def visit(ozel, node, context):
		method_name = f'visit_{type(node).__name__}'
		method = getattr(ozel, method_name, ozel.no_visit_method)
		return method(node, context)

	def no_visit_method(ozel, node, context):
		raise Exception(f'No visit_{type(node).__name__} method defined')

	###################################

	def visit_NumberNode(ozel, node, context):
		return RTResult().success(
			Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
		)

	def visit_BinOpNode(ozel, node, context):
		res = RTResult()
		left = res.register(ozel.visit(node.left_node, context))
		if res.error: return res
		right = res.register(ozel.visit(node.right_node, context))
		if res.error: return res

		if node.op_tok.type == TT_PLUS:
			result, error = left.added_to(right)
		elif node.op_tok.type == TT_MINUS:
			result, error = left.subbed_by(right)
		elif node.op_tok.type == TT_MUL:
			result, error = left.multed_by(right)
		elif node.op_tok.type == TT_DIV:
			result, error = left.dived_by(right)

		if error:
			return res.failure(error)
		else:
			return res.success(result.set_pos(node.pos_start, node.pos_end))

	def visit_UnaryOpNode(ozel, node, context):
		res = RTResult()
		number = res.register(ozel.visit(node.node, context))
		if res.error: return res

		error = None

		if node.op_tok.type == TT_MINUS:
			number, error = number.multed_by(Number(-1))

		if error:
			return res.failure(error)
		else:
			return res.success(number.set_pos(node.pos_start, node.pos_end))

#######################################
# RUN
#######################################

def run(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error

	# Run program
	interpreter = Interpreter()
	context = Context('<program>')
	result = interpreter.visit(ast.node, context)

	return result.value, result.error
