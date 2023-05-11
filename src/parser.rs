use sexp::Atom::*;
use sexp::*;

use crate::constants::*;

/**
 * Reserved words that cannot be used for identifier names
 */
const RESERVED_WORDS: &'static [&str] = &["true", "false","add1", "sub1", "isnum", "isbool","let", "block", "set", "if", "break", "set!", "loop","+", "-", "*", "=", ">", ">=", "<", "<=", "input", "print"];

/**
 * Converts the contents of the input program into a Program AST following the abstract syntax
 * Panics if the input program cannot be converted into a Sexpr AST
 */
pub fn parse_program(input_program_contents: &str) -> Program {
    let input_program_contents_as_list = "(".to_owned() + &input_program_contents + ")";
    let file_contents_as_sexpr = &parse(&input_program_contents_as_list);
    match file_contents_as_sexpr {
        Ok(s) => parse_program_into_ast(s),
        Err(_) => panic!("Invalid unable to parse input file into valid SExpr AST."),
    }
}

/**
 * Converts the Sexpr AST into a valid Program struct for the compiler
 * Panics if the AST doesn't follow the Abstract Syntax
 */
fn parse_program_into_ast(s: &Sexp) -> Program {
    match s {
        Sexp::List(vec) => {
            let mut defs: Vec<Definition> = vec![];
            for def_or_exp in vec {
                if is_function_definition(def_or_exp) {
                    defs.push(parse_sexp_into_func_def(def_or_exp));
                } else {
                    return Program {
                        defs: defs,
                        main: parse_sexp_into_expr(def_or_exp),
                    };
                }
            }
            panic!("Invalid Only found definitions");
        }
        _ => panic!("Invalid Program should be a list")
    }
}

/**
 * Checks if an Sexp corresponds to a function definition
 */
fn is_function_definition(s: &Sexp) -> bool {
    match s {
        Sexp::List(vec) => match &vec[..] {
            [Sexp::Atom(S(keyword)), Sexp::List(_), _] if keyword == "fun" => true,
            _ => false
        },
        _ => false
    }
}

/**
 * Converts an Sexp corresponding to a function definition into a compilable AST
 * Panics if the function definition doesn't follow the abstract syntax
 */
fn parse_sexp_into_func_def(s: &Sexp) -> Definition {
    match s {
        Sexp::List(def_vec) => match &def_vec[..] {
            [Sexp::Atom(S(keyword)), Sexp::List(name_vec), body] if keyword == "fun" => match name_vec.len() {
                0 => panic!("Invalid Cannot create a function definiton without a function name."),
                _ => {
                    match name_vec.len() {
                        1 => Definition::FunNoArg(name_vec[0].to_string(), parse_sexp_into_expr(body)),
                        _ => Definition::FunWithArg(name_vec[0].to_string(), name_vec.iter().skip(0).map(|name_as_sexpr| name_as_sexpr.to_string()).collect(), parse_sexp_into_expr(body))
                    }
                }
            },
            _ => panic!("Invalid Not a function definition"),
        },
        _ => panic!("Invalid Not a function definition"),
    }
}

/**
 * Converts an Sexp corresponding to an expression into a compilable AST
 */
fn parse_sexp_into_expr(s: &Sexp) -> Expr {
    match s {
        Sexp::Atom(s) => parse_sexpr_atom(s),
        Sexp::List(vec) => {
            match &vec[..] {
                [Sexp::Atom(S(op)), e] if op == "add1" => parse_unop_expr(Op1::Add1, e),
                [Sexp::Atom(S(op)), e] if op == "sub1" => parse_unop_expr(Op1::Sub1, e),
                [Sexp::Atom(S(op)), e] if op == "isnum" => parse_unop_expr(Op1::IsNum, e),
                [Sexp::Atom(S(op)), e] if op == "isbool" => parse_unop_expr(Op1::IsBool, e),
                [Sexp::Atom(S(op)), e] if op == "print" => parse_unop_expr(Op1::Print, e),
                [Sexp::Atom(S(op)), e1, e2] if op == "+" => parse_binop_expr(Op2::Plus, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == "-" => parse_binop_expr(Op2::Minus, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == "*" => parse_binop_expr(Op2::Times, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == "<" => parse_binop_expr(Op2::Less, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == ">" => parse_binop_expr(Op2::Greater, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == "=" => parse_binop_expr(Op2::Equal, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == "<=" => parse_binop_expr(Op2::LessEqual, e1, e2),
                [Sexp::Atom(S(op)), e1, e2] if op == ">=" => parse_binop_expr(Op2::GreaterEqual, e1, e2),
                [Sexp::Atom(S(op)),Sexp::List(bindings), body] if op == "let" => parse_let_expr(bindings, body),
                [Sexp::Atom(S(op)), Sexp::Atom(S(id)), e] if op == "set" => parse_set_expr(id, e),
                [Sexp::Atom(S(op)), cond, then, els] if op == "if" =>  parse_if_expr(cond, then, els),
                [Sexp::Atom(S(op)), exprs @ ..] if op == "block" => parse_block_operation(exprs),
                [Sexp::Atom(S(op)), e] if op == "loop" => parse_loop_expr(e),
                [Sexp::Atom(S(op)), e] if op == "break" => parse_break_expr(e),
                [Sexp::Atom(S(funcname))] => parse_call_no_args(funcname),
                [Sexp::Atom(S(funcname)), Sexp::List(params)] => parse_call(funcname, params),
                _ => panic!("Invalid Sexpr format: {s}")
            }
        },
    }
}

/**
 * Parse unary operator expressions
 */
fn parse_unop_expr(op: Op1, e: &Sexp) -> Expr {
    Expr::UnOp(op, Box::new(parse_sexp_into_expr(e)))
}

/**
 * Parse binary operator expressions
 */
fn parse_binop_expr(op: Op2, e1: &Sexp, e2: &Sexp) -> Expr {
    Expr::BinOp(op, Box::new(parse_sexp_into_expr(e1)), Box::new(parse_sexp_into_expr(e2)))
}

/**
 * Parse Let expressions
 * Panics if binding list is empty
 */
fn parse_let_expr(bindings: &Vec<Sexp>, body: &Sexp) -> Expr {
    if bindings.is_empty() {
        panic!("Invalid let without binding")
    }
    let mut vec = Vec::new();
    for binding in bindings {
        vec.push(parse_bind(binding));
    }

    Expr::Let(vec, Box::new(parse_sexp_into_expr(body)))
}

/**
 * Parses the bindings of a let expression
 * Panics if reserved word is used for identifier, and binding follows invalid format
 */
fn parse_bind(sexp: &Sexp) -> (String, Expr) {
    match sexp {
        Sexp::List(vec) => match &vec[..] {
            [Sexp::Atom(S(s)), e] => {
                if RESERVED_WORDS.contains(&&s[..]) {
                    panic!("Invalid identifier name: {s} since it is a keyword.");
                } else {
                   (s.to_string(), parse_sexp_into_expr(e))
                }
            },
            _ => panic!("Invalid binding format"),
        }
        _ => panic!("Invalid let binding"),
    }
  }

/**
 * Parse If expressions
 */
fn parse_if_expr(cond: &Sexp, then: &Sexp, els: &Sexp) -> Expr {
    Expr::If(Box::new(parse_sexp_into_expr(cond)), Box::new(parse_sexp_into_expr(then)), Box::new(parse_sexp_into_expr(els)))
}

/**
 * Parse Set expressions
 * Panics if a reserved word is being used as an identifier name
 */
fn parse_set_expr(id: &str, e: &Sexp) -> Expr {
    match RESERVED_WORDS.contains(&id) {
        true => panic!("Invalid identifier in set operation: {id}, which is a keyword"),
        false => Expr::Set(id.to_string(), Box::new(parse_sexp_into_expr(e)))
    }
}

/**
 * Parse Loop expressions
 */
fn parse_loop_expr(e: &Sexp) -> Expr {
    Expr::Loop(Box::new(parse_sexp_into_expr(e)))
}

/**
 * Parse Block expressions
 * Panics if block is empty
 */
fn parse_block_operation(exprs: &[Sexp]) -> Expr {
    match exprs.is_empty() {
        true => panic!("Invalid empty block"),
        false => Expr::Block(exprs.iter().map(|expression| parse_sexp_into_expr(expression)).collect())
    }
}

/**
 * Parse Break Expression
 */
fn parse_break_expr(e: &Sexp) -> Expr {
    Expr::Break(Box::new(parse_sexp_into_expr(e)))
}
/**
 * Converts an Sexp Atom into a lead node of the compilable AST
 * Panics if an unsupported type is provided in the input program (not a string or num)
 */
fn parse_sexpr_atom(s: &Atom) -> Expr {
    match s {
        S(bool) if bool == "true" => Expr::Boolean(true),
        S(bool) if bool == "false" => Expr::Boolean(false),
        S(s) => parse_id(s),
        I(n) => parse_num(n),
        _ => panic!("Invalid unsupported type."),
    }
}

/**
 * Parses function call expression for no arg case
 */
fn parse_call_no_args(funcname: &String) -> Expr {
    if RESERVED_WORDS.contains(&&funcname[..]) {
        panic!("Invalid usage of expression: {funcname}")
    }
    Expr::CallNoArg(funcname.to_string())
}

/**
 * Parses function call expression
 */
fn parse_call(funcname: &String, args: &Vec<Sexp>) -> Expr {
    if RESERVED_WORDS.contains(&&funcname[..]) {
        panic!("Invalid usage of expression: {funcname}")
    }
    Expr::Call(funcname.to_string(), args.into_iter().map(|sexp| parse_sexp_into_expr(sexp)).collect())
}

/**
 * Converts an identifier into an Expr object for the AST
 * Panics if a reserved word is being used for the identifier
 */
fn parse_id(s: &str) -> Expr {
    match s == "input" || !RESERVED_WORDS.contains(&s) {
        true => Expr::Id(s.to_string()),
        false => panic!("Invalid identifier name since it is a reserved word: {s}"),
    }
}

/**
 * Converts a number into an Expr object for the AST
 * Panics if the number is out of 63-bit range (signed)
 */
fn parse_num(n: &i64) -> Expr {
    match *n > INT_MAX || *n < INT_MIN {
        true => panic!("Invalid overflow: {n}"),
        false => Expr::Number(i64::try_from(*n).unwrap()),
    }
}