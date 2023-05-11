use std::collections::HashSet;

use im::HashMap;

use crate::constants::*;
use crate::error::*;

pub const ERROR_LABEL: &str = "throw_error";

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
    IJne(String),
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
    for def in &program.defs[..] {
        defs.push_str(&compile_definition(&def, &mut labels));
    }
    let main = compile_expr_to_string(
        &program.main,
        &ContextBuilder::new(2, &HashMap::new(), &String::from("")).build(),
        &mut labels,
    );
    (defs, main)
}

fn compile_definition(d: &Definition, labels: &mut i32) -> String {
    match d {
        Definition::FunNoArg(_, _) => todo!(),
        Definition::FunWithArg(_, _, _) => todo!(),
    }
}

fn compile_expr_to_string(e: &Expr, context: &Context, labels: &mut i32) -> String {
    let mut instrs: Vec<Instr> = Vec::new();
    compile_to_instrs(e, &mut instrs, context, labels);
    /* print!("{}", instrs.iter().map(instr_to_str).collect::<String>()); */

    instrs.iter().map(instr_to_str).collect::<String>()
}

fn compile_to_instrs(expr: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i32) {
    match expr {
        Expr::Number(n) => instrs.push(Instr::IMov(Val::Reg(RAX), convert_i64_to_val(n))),
        Expr::Boolean(b) => instrs.push(Instr::IMov(Val::Reg(RAX), convert_bool_to_val(b))),
        Expr::Id(s) if s == "input" => instrs.push(Instr::IMov(Val::Reg(RAX), Val::Reg(RDI))),
        Expr::Id(id) => match context.env.contains_key(id) {
            true => instrs.push(Instr::IMov(
                Val::Reg(RAX),
                Val::RegOffset(RSP, *(context.env.get(id).unwrap())),
            )),
            false => panic!("Unbound variable identifier {id}"),
        },

        Expr::Let(bindings, body) => compile_let(bindings, body, instrs, context, labels),
        Expr::UnOp(op, e) => compile_unop(op, e, instrs, context, labels),
        Expr::BinOp(op, e1, e2) => compile_binop(op, e1, e2, instrs, context, labels),
        Expr::Set(id, e) => compile_set(id, e, instrs, context, labels),
        Expr::If(cond, then, els) => compile_if(cond, then, els, instrs, context, labels),
        Expr::Block(es) => es.iter().for_each(|e| compile_to_instrs(e, instrs, context, labels)),
        Expr::Loop(e) => compile_loop(e, instrs, context, labels),
        Expr::Break(e) => compile_break(e, instrs, context, labels),
        Expr::CallNoArg(_) => todo!(),
        Expr::Call(_, _) => todo!(),
    }
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
            Val::RegOffset(RSP, *(context.env.get(id).unwrap())),
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
    match op {
        Op2::Plus => {
            compile_to_instrs(e1, instrs, context, labels);
            instrs.push(Instr::IMov(Val::RegOffset(RSP, context.si), Val::Reg(RAX)));
            compile_to_instrs(
                e2,
                instrs,
                &ContextBuilder::from(context).si(context.si + 1).build(),
                labels,
            );
            check_if_both_num(instrs, Val::RegOffset(RSP, context.si), Val::Reg(RAX));
            instrs.push(Instr::IAdd(Val::Reg(RAX), Val::RegOffset(RSP, context.si)));
            check_for_overflow(instrs);
        }
        Op2::Minus => {
            compile_to_instrs(e2, instrs, context, labels);
            instrs.push(Instr::IMov(Val::RegOffset(RSP, context.si), Val::Reg(RAX)));
            compile_to_instrs(
                e1,
                instrs,
                &ContextBuilder::from(context).si(context.si + 1).build(),
                labels,
            );
            check_if_both_num(instrs, Val::RegOffset(RSP, context.si), Val::Reg(RAX));
            instrs.push(Instr::IAdd(Val::Reg(RAX), Val::RegOffset(RSP, context.si)));
            check_for_overflow(instrs);
        }
        Op2::Times => {
            compile_to_instrs(e1, instrs, context, labels);
            instrs.push(Instr::IMov(Val::RegOffset(RSP, context.si), Val::Reg(RAX)));
            compile_to_instrs(
                e2,
                instrs,
                &ContextBuilder::from(context).si(context.si + 1).build(),
                labels,
            );
            check_if_both_num(instrs, Val::RegOffset(RSP, context.si), Val::Reg(RAX));
            instrs.push(Instr::ISar(Val::Reg(RAX), Val::Imm(1)));
            instrs.push(Instr::IMul(Val::Reg(RAX), Val::RegOffset(RSP, context.si)));
            check_for_overflow(instrs);
        }
        _ => compile_conditional_binop(op, e1, e2, instrs, context, labels),
    }
}

fn compile_conditional_binop(
    op: &Op2,
    e1: &Expr,
    e2: &Expr,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i32,
) {
    compile_to_instrs(e2, instrs, context, labels);
    instrs.push(Instr::IMov(Val::RegOffset(RSP, context.si), Val::Reg(RAX)));
    compile_to_instrs(
        e1,
        instrs,
        &ContextBuilder::from(context).si(context.si + 1).build(),
        labels,
    );
    match op {
        Op2::Equal => check_if_same_type(instrs, Val::Reg(RAX), Val::RegOffset(RSP, context.si)),
        _ => check_if_both_num(instrs, Val::Reg(RAX), Val::RegOffset(RSP, context.si)),
    };
    instrs.push(Instr::ICmp(Val::Reg(RAX), Val::RegOffset(RSP, context.si)));
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
        Op1::Print => todo!(),
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
        instrs.push(Instr::IMov(Val::RegOffset(RSP, new_si), Val::Reg(RAX)));
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
        Instr::IJne(s) => format!("\njne {}", s.to_string()),
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
    }
}

// NOTE: we need to multiply si by 8 when creating the final x86 instructions
fn val_to_str(v: &Val) -> String {
    match v {
        Val::Imm(n) => (*n).to_string(),
        Val::Reg(r) => reg_to_str(r),
        Val::RegOffset(r, n) => format!("[{}-{}]", reg_to_str(r), n * 8),
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
