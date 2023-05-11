/**
 * 63-bit Int Max
 */
pub const INT_MAX: i64 = 4611686018427387903;

/**
 * 63-bit Int MIN
 */
pub const INT_MIN: i64 = -4611686018427387904;

/**
 * Abstract Syntax for the Program
 */
#[derive(Debug)]
pub struct Program {
    pub defs: Vec<Definition>,
    pub main: Expr,
}

/**
 * Abstract Syntax for a Function Definition
 * (Function Name, <Arg Names>, Body)
 */
#[derive(Debug)]
pub enum Definition {
    FunNoArg(String, Expr),
    FunWithArg(String, Vec<String>, Expr)
}

/**
 * Abstract Syntax for an Expression
 */
#[derive(Debug)]
pub enum Expr {
    Number(i64),
    Boolean(bool),
    Id(String),
    Let(Vec<(String, Expr)>, Box<Expr>),
    UnOp(Op1, Box<Expr>),
    BinOp(Op2, Box<Expr>, Box<Expr>),
    Set(String, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Block(Vec<Expr>),
    Loop(Box<Expr>),
    Break(Box<Expr>),
    CallNoArg(String),
    Call(String, Vec<Expr>)
}

/**
 * Abstract Syntax for Unary operations
 */
#[derive(Debug)]
pub enum Op1 {
    Add1,
    Sub1,
    IsNum,
    IsBool,
    Print
}

/**
 * Abstract Syntax for Binary operations
 */
#[derive(Debug)]
pub enum Op2 {
    Plus,
    Minus,
    Times,
    Equal, 
    Greater, 
    GreaterEqual, 
    Less, 
    LessEqual
}