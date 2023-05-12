use std::collections::HashSet;

use im::HashMap;

use crate::constants::*;
use crate::error::*;

pub const ERROR_LABEL: &str = "throw_error";
pub const PRINT_LABEL: &str = "snek_print";

#[derive(Debug)]
enum Val {
    Reg(Reg),
    Imm(i64),
    RegOffset(Reg, i32),
}

#[derive(Debug)]
enum VarTypes {
    NUM,
    BOOL,
}

#[derive(Debug)]
enum Reg {
    RAX,
    RSP,
    RDI,
    RBX,
}

use Reg::*;

#[derive(Debug)]
enum Instr {
    IMov(Val, Val),
    IAdd(Val, Val),
    ISub(Val, Val),
    IMul(Val, Val),
    ITest(Val, Val),
    IXor(Val, Val),
    ISar(Val, Val),
    ICmp(Val, Val),
    IJe(String),
    IJo(String),
    IJmp(String),
    IJnz(String),
    ILabel(String),
    ICmove(Val, Val),
    ICmovz(Val, Val),
    ICmovl(Val, Val),
    ICmovle(Val, Val),
    ICmovg(Val, Val),
    ICmovge(Val, Val),
    IOr(Val, Val),
    IAnd(Val, Val),
    ICall(String),
}
struct Context<'a> {
    si: i32,
    env: &'a HashMap<String, i32>,
    brake: &'a str,
}

// Prompted Chatgpt to implement a builder class for the Context struct using the prompt:
// "Generate a builder class for the following struct <Insert My Context Implementation Here>"

struct ContextBuilder<'a> {
    si: i32,
    env: Option<&'a HashMap<String, i32>>,
    brake: Option<&'a str>,
}

impl<'a> ContextBuilder<'a> {
    fn new(si: i32, env: &'a HashMap<String, i32>, brake: &'a str) -> Self {
        Self {
            si,
            env: Some(env),
            brake: Some(brake),
        }
    }

    fn env(mut self, env: &'a HashMap<String, i32>) -> Self {
        self.env = Some(env);
        self
    }

    fn brake(mut self, brake: &'a str) -> Self {
        self.brake = Some(brake);
        self
    }

    fn si(mut self, si: i32) -> Self {
        self.si = si;
        self
    }

    fn from(context: &'a Context<'a>) -> Self {
        Self {
            si: context.si,
            env: Some(context.env),
            brake: Some(context.brake),
        }
    }

    fn build(self) -> Context<'a> {
        Context {
            si: self.si,
            env: self.env.expect("env is not set"),
            brake: self.brake.expect("brake is not set"),
        }
    }
}

pub fn compile_program(program: &Program) -> (String, String) {
    let mut labels: i32 = 0;
    let mut defs: String = String::new();

    if duplicate_func_names(&program.defs) {
        panic!("Invalid duplicate func name")
    }
    for def in &program.defs[..] {
        defs.push_str(&compile_definition(&def, &mut labels));
    }
    let main = compile_expr_to_string(
        &program.main,
        &ContextBuilder::new(0, &HashMap::new(), &String::from("")).build(),
        &mut labels,
    );
    (defs, main)
}

fn compile_definition(d: &Definition, labels: &mut i32) -> String {
    match d {
        Definition::Func(funcname, args, body) => {
            let mut env = HashMap::new();
            let mut loc = 1;
            let depth = depth(body);
/*             println!("{:?}", d); */
            for arg in args {
/*                 println!("arg: {} depth: {} loc: {}", arg, depth, loc); */
                env.insert(arg.to_string(), 8*depth + 8*loc);
                loc += 1;
            }
            let body_instrs_as_str = compile_expr_to_string(
                body,
                &ContextBuilder::new(0, &env, &String::from("")).build(),
                labels,
            );
            format!(
                "{funcname}:
                {body_instrs_as_str}
                ret
                "
            )
        }
    }
}

// Generated lambda for Expr::Call calculation using ChatGPT by providing it a for loop implementation and prompting it to simplify the code
fn depth(e: &Expr) -> i32 {
    match e {
        Expr::Number(_) => 0,
        Expr::Boolean(_) => 0,
        Expr::UnOp(_, expr) => depth(expr),
        Expr::BinOp(_, e1, e2) => depth(e2).max(depth(e1) + 1), // For binops we compile RHS first for ease with Minus and Conditionals
        Expr::Let(bindings, body) => bindings
            .iter()
            .enumerate()
            .map(|(index, (_, expr))| depth(expr) + index as i32)
            .max()
            .unwrap_or(0)
            .max(depth(body) + bindings.len() as i32),
        Expr::Id(_) => 0,
        Expr::If(expr1, expr2, expr3) => depth(expr1).max(depth(expr2)).max(depth(expr3)),
        Expr::Loop(expr) => depth(expr),
        Expr::Block(exprs) => exprs.iter().map(|expr| depth(expr)).max().unwrap_or(0),
        Expr::Break(expr) => depth(expr),
        Expr::Set(_, expr) => depth(expr),
        Expr::Call(_, arg_exprs) => arg_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| depth(expr) + index as i32)
            .max()
            .unwrap_or(0),
    }
}

fn compile_expr_to_string(e: &Expr, context: &Context, labels: &mut i32) -> String {
    let depth = depth(e);
    let offset = depth * 8;
    let mut instrs: Vec<Instr> = Vec::new();
    compile_to_instrs(e, &mut instrs, context, labels);
    /* print!("{}", instrs.iter().map(instr_to_str).collect::<String>()); */
    let instrs_as_str = instrs.iter().map(instr_to_str).collect::<String>();
    format!(
        "
        sub rsp, {offset}
        {instrs_as_str}
        add rsp, {offset}
        "
    )
}

fn compile_to_instrs(expr: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i32) {
    match expr {
        Expr::Number(n) => instrs.push(Instr::IMov(Val::Reg(RAX), convert_i64_to_val(n))),
        Expr::Boolean(b) => instrs.push(Instr::IMov(Val::Reg(RAX), convert_bool_to_val(b))),
        Expr::Id(s) if s == "input" => instrs.push(Instr::IMov(Val::Reg(RAX), Val::Reg(RDI))),
        Expr::Id(id) => match context.env.contains_key(id) {
            true => instrs.push(Instr::IMov(
                Val::Reg(RAX),
                Val::RegOffset(RSP, *context.env.get(id).unwrap()),
            )),
            false => panic!("Unbound variable identifier {id}"),
        },

        Expr::Let(bindings, body) => compile_let(bindings, body, instrs, context, labels),
        Expr::UnOp(op, e) => compile_unop(op, e, instrs, context, labels),
        Expr::BinOp(op, e1, e2) => compile_binop(op, e1, e2, instrs, context, labels),
        Expr::Set(id, e) => compile_set(id, e, instrs, context, labels),
        Expr::If(cond, then, els) => compile_if(cond, then, els, instrs, context, labels),
        Expr::Block(es) => es
            .iter()
            .for_each(|e| compile_to_instrs(e, instrs, context, labels)),
        Expr::Loop(e) => compile_loop(e, instrs, context, labels),
        Expr::Break(e) => compile_break(e, instrs, context, labels),
        Expr::Call(funcname, arg_exprs) => {
            compile_func_call(funcname, arg_exprs, instrs, context, labels)
        }
    }
}

fn compile_func_call(
    funcname: &String,
    arg_exprs: &Vec<Expr>,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i32,
) {
    let mut offset = 0;
    let mut new_si = context.si;
    if arg_exprs.len() > 0 {
        for arg_expr in arg_exprs.iter().take(arg_exprs.len()-1) {
            compile_to_instrs(arg_expr, instrs, &ContextBuilder::from(context).si(new_si).build(), labels);
            instrs.push(Instr::IMov(Val::RegOffset(RSP, offset), Val::Reg(RAX)));
            offset += 8;
            new_si = new_si + 1;
        }
        compile_to_instrs(arg_exprs.get(arg_exprs.len()-1).unwrap(), instrs, &ContextBuilder::from(context).si(new_si).build(), labels);
        offset += 8;
    }
    // for rdi
    offset += 8;
    instrs.push(Instr::ISub(Val::Reg(RSP), Val::Imm(offset.into())));
    // Move args in stack to be near new RSP location
    let mut cur_offset_to_old_loc = offset;
    let mut cur_offset_to_new_loc = 0;
    if arg_exprs.len() > 0 {
        for _ in arg_exprs.iter().take(arg_exprs.len()-1) {
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::RegOffset(RSP, cur_offset_to_old_loc)));
            instrs.push(Instr::IMov(Val::RegOffset(RSP, cur_offset_to_new_loc), Val::Reg(RBX)));
            cur_offset_to_old_loc+=8;
            cur_offset_to_new_loc+=8;
        }
        instrs.push(Instr::IMov(Val::RegOffset(RSP, cur_offset_to_new_loc), Val::Reg(RAX)));
        cur_offset_to_new_loc+=8;
    }
    instrs.push(Instr::IMov(Val::RegOffset(RSP, cur_offset_to_new_loc), Val::Reg(RDI)));
    // Call
    instrs.push(Instr::ICall(funcname.to_string()));
    // Restore rdi
    instrs.push(Instr::IMov(Val::Reg(RDI), Val::RegOffset(RSP, cur_offset_to_new_loc)));
    // Restore stack pointer
    instrs.push(Instr::IAdd(Val::Reg(RSP), Val::Imm(offset.into())));
}

fn compile_break(e: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i32) {
    if context.brake == "" {
        panic!("Error: break occurred outside a loop in sexp")
    }
    compile_to_instrs(e, instrs, context, labels);
    instrs.push(Instr::IJmp(context.brake.to_string()));
}

fn compile_loop(e: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i32) {
    let startloop = new_label(labels, "loop");
    let endloop = new_label(labels, "loopend");
    instrs.push(Instr::ILabel(startloop.to_string()));
    compile_to_instrs(
        e,
        instrs,
        &ContextBuilder::from(context).brake(&endloop).build(),
        labels,
    );
    instrs.push(Instr::IJmp(startloop.to_string()));
    instrs.push(Instr::ILabel(endloop.to_string()));
}

fn compile_if(
    cond: &Expr,
    then: &Expr,
    els: &Expr,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i32,
) {
    let end_label = new_label(labels, "ifend");
    let else_label = new_label(labels, "ifelse");

    compile_to_instrs(cond, instrs, context, labels);
    instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(1)));
    instrs.push(Instr::IJe(else_label.to_string()));

    compile_to_instrs(then, instrs, context, labels);
    instrs.push(Instr::IJmp(end_label.to_string()));

    instrs.push(Instr::ILabel(else_label.to_string()));
    compile_to_instrs(els, instrs, context, labels);

    instrs.push(Instr::ILabel(end_label.to_string()));
}

fn compile_set(
    id: &String,
    e: &Expr,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i32,
) {
    if !context.env.contains_key(id) {
        panic!("Unbound variable identifier {id}")
    } else {
        compile_to_instrs(e, instrs, context, labels);
        instrs.push(Instr::IMov(
            Val::RegOffset(RSP, 8 * (context.env.get(id).unwrap())),
            Val::Reg(RAX),
        ));
    }
}

fn compile_binop(
    op: &Op2,
    e1: &Expr,
    e2: &Expr,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i32,
) {
    compile_to_instrs(e2, instrs, context, labels);
    instrs.push(Instr::IMov(
        Val::RegOffset(RSP, 8 * context.si),
        Val::Reg(RAX),
    ));
    compile_to_instrs(
        e1,
        instrs,
        &ContextBuilder::from(context).si(context.si + 1).build(),
        labels,
    );
    match op {
        Op2::Plus => {
            check_if_both_num(instrs, Val::RegOffset(RSP, 8 * context.si), Val::Reg(RAX));
            instrs.push(Instr::IAdd(
                Val::Reg(RAX),
                Val::RegOffset(RSP, 8 * context.si),
            ));
            check_for_overflow(instrs);
        }
        Op2::Minus => {
            check_if_both_num(instrs, Val::RegOffset(RSP, 8 * context.si), Val::Reg(RAX));
            instrs.push(Instr::ISub(
                Val::Reg(RAX),
                Val::RegOffset(RSP, 8 * context.si),
            ));
            check_for_overflow(instrs);
        }
        Op2::Times => {
            check_if_both_num(instrs, Val::RegOffset(RSP, 8 * context.si), Val::Reg(RAX));
            instrs.push(Instr::ISar(Val::Reg(RAX), Val::Imm(1)));
            instrs.push(Instr::IMul(
                Val::Reg(RAX),
                Val::RegOffset(RSP, 8 * context.si),
            ));
            check_for_overflow(instrs);
        }
        _ => compile_conditional_binop(op, instrs, context),
    }
}

fn compile_conditional_binop(
    op: &Op2,
    instrs: &mut Vec<Instr>,
    context: &Context
) {
    match op {
        Op2::Equal => {
            check_if_same_type(instrs, Val::Reg(RAX), Val::RegOffset(RSP, 8 * context.si))
        }
        _ => check_if_both_num(instrs, Val::Reg(RAX), Val::RegOffset(RSP, 8 * context.si)),
    };
    instrs.push(Instr::ICmp(
        Val::Reg(RAX),
        Val::RegOffset(RSP, 8 * context.si),
    ));
    instrs.push(Instr::IMov(Val::Reg(RAX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(3)));
    match op {
        Op2::Equal => instrs.push(Instr::ICmove(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::Greater => instrs.push(Instr::ICmovg(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::GreaterEqual => instrs.push(Instr::ICmovge(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::Less => instrs.push(Instr::ICmovl(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::LessEqual => instrs.push(Instr::ICmovle(Val::Reg(RAX), Val::Reg(RBX))),
        _ => unreachable!(),
    }
}

fn compile_unop(op: &Op1, e: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i32) {
    compile_to_instrs(e, instrs, context, labels);
    match op {
        Op1::Add1 => {
            check_if_num(instrs);
            instrs.push(Instr::IAdd(Val::Reg(RAX), Val::Imm(2)));
            check_for_overflow(instrs);
        }
        Op1::Sub1 => {
            check_if_num(instrs);
            instrs.push(Instr::ISub(Val::Reg(RAX), Val::Imm(2)));
            check_for_overflow(instrs);
        }
        Op1::IsNum => check_type(instrs, VarTypes::NUM),
        Op1::IsBool => check_type(instrs, VarTypes::BOOL),
        Op1::Print => {
            let offset = 8 * (context.si + 1 + (context.si % 2));
            instrs.push(Instr::ISub(Val::Reg(RSP), Val::Imm(offset.into())));
            instrs.push(Instr::IMov(Val::RegOffset(RSP, 0), Val::Reg(RDI)));
            instrs.push(Instr::IMov(Val::Reg(RDI), Val::Reg(RAX)));
            instrs.push(Instr::ICall(PRINT_LABEL.to_string()));
            instrs.push(Instr::IMov(Val::Reg(RDI), Val::RegOffset(RSP, 0)));
            instrs.push(Instr::IAdd(Val::Reg(RSP), Val::Imm(offset.into())));
        }
    }
}

// Converted loop in cobra implementation to new lambda using ChatGPT with prompt:
// "Given this loop <INSERT MY LOOP CODE HERE> generate a lambda"
fn compile_let(
    bindings: &Vec<(String, Expr)>,
    body: &Expr,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i32,
) {
    if duplicate_bindings(bindings) {
        panic!("Duplicate binding")
    }

    let mut new_env = context.env.clone();
    let mut new_si = context.si;

    bindings.iter().for_each(|(identifier, binding_expr)| {
        compile_to_instrs(
            &binding_expr,
            instrs,
            &ContextBuilder::from(context)
                .env(&new_env)
                .si(new_si)
                .build(),
            labels,
        );
        instrs.push(Instr::IMov(Val::RegOffset(RSP, 8 * new_si), Val::Reg(RAX)));
        new_env = new_env.update(identifier.to_string(), new_si);
        new_si += 1;
    });

    compile_to_instrs(
        body,
        instrs,
        &ContextBuilder::from(context)
            .env(&new_env)
            .si(new_si)
            .build(),
        labels,
    );
}

fn duplicate_bindings(bindings: &Vec<(String, Expr)>) -> bool {
    let mut unique_var_identifiers = HashSet::new();
    return bindings
        .iter()
        .any(|binding| unique_var_identifiers.insert(binding.0.to_string()) == false);
}

fn duplicate_func_names(definitions: &Vec<Definition>) -> bool {
    let mut unique_func_name = HashSet::new();
    for definition in definitions {
        let Definition::Func(name, _, _) = definition;
        if unique_func_name.insert(name) == false {
            return true;
        }
    }
    return false;
}

fn convert_i64_to_val(n: &i64) -> Val {
    let deref = *n;
    if deref > INT_MAX || deref < INT_MIN {
        panic!("Invalid overflow: {n}")
    } else {
        Val::Imm(*n << 1)
    }
}

fn convert_bool_to_val(b: &bool) -> Val {
    match b {
        true => Val::Imm(3),
        false => Val::Imm(1),
    }
}

fn new_label(l: &mut i32, s: &str) -> String {
    let current = *l;
    *l += 1;
    format!("{s}_{current}")
}

fn check_if_num(instrs: &mut Vec<Instr>) {
    instrs.push(Instr::ITest(Val::Reg(RAX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJnz(ERROR_LABEL.to_string()));
}

fn check_if_both_num(instrs: &mut Vec<Instr>, r1: Val, r2: Val) {
    instrs.push(Instr::IMov(Val::Reg(RBX), r1));
    instrs.push(Instr::IOr(Val::Reg(RBX), r2));
    instrs.push(Instr::IAnd(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::ITest(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJnz(ERROR_LABEL.to_string()));
}

fn check_if_same_type(instrs: &mut Vec<Instr>, r1: Val, r2: Val) {
    instrs.push(Instr::IMov(Val::Reg(RBX), r1));
    instrs.push(Instr::IXor(Val::Reg(RBX), r2));
    instrs.push(Instr::IAnd(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::ITest(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJnz(ERROR_LABEL.to_string()));
}

fn check_for_overflow(instrs: &mut Vec<Instr>) {
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OVERFLOW_ERROR_CODE)));
    instrs.push(Instr::IJo(ERROR_LABEL.to_string()));
}

fn check_type(instrs: &mut Vec<Instr>, type_to_check: VarTypes) {
    instrs.push(Instr::ITest(Val::Reg(RAX), Val::Imm(1)));
    match type_to_check {
        VarTypes::NUM => {
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(3)));
            instrs.push(Instr::IMov(Val::Reg(RAX), Val::Imm(1)));
        }
        VarTypes::BOOL => {
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(1)));
            instrs.push(Instr::IMov(Val::Reg(RAX), Val::Imm(3)));
        }
    }
    instrs.push(Instr::ICmovz(Val::Reg(RAX), Val::Reg(RBX)))
}
fn instr_to_str(i: &Instr) -> String {
    match i {
        Instr::IMov(v1, v2) => format!("\nmov {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IAdd(v1, v2) => format!("\nadd {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IMul(v1, v2) => format!("\nimul {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ISub(v1, v2) => format!("\nsub {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ITest(v1, v2) => format!("\ntest {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IXor(v1, v2) => format!("\nxor {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IOr(v1, v2) => format!("\nor {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IAnd(v1, v2) => format!("\nand {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmp(v1, v2) => format!("\ncmp {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IJe(s) => format!("\nje {}", s.to_string()),
        Instr::IJmp(s) => format!("\njmp {}", s.to_string()),
        Instr::ILabel(s) => format!("\n {}:", s.to_string()),
        Instr::IJo(s) => format!("\njo {}", s.to_string()),
        Instr::IJnz(s) => format!("\njnz {}", s.to_string()),
        Instr::ISar(v1, v2) => format!("\nsar {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmove(v1, v2) => format!("\ncmove {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmovz(v1, v2) => format!("\ncmovz {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmovl(v1, v2) => format!("\ncmovl {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmovle(v1, v2) => format!("\ncmovle {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmovg(v1, v2) => format!("\ncmovg {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmovge(v1, v2) => format!("\ncmovge {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICall(s) => format!("\ncall {}", s.to_string()),
    }
}

fn val_to_str(v: &Val) -> String {
    match v {
        Val::Imm(n) => (*n).to_string(),
        Val::Reg(r) => reg_to_str(r),
        Val::RegOffset(r, n) => format!("[{}+{}]", reg_to_str(r), n),
    }
}

fn reg_to_str(r: &Reg) -> String {
    match r {
        RAX => "rax".to_string(),
        RSP => "rsp".to_string(),
        RDI => "rdi".to_string(),
        RBX => "rbx".to_string(),
    }
}
