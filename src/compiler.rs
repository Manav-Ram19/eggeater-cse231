use std::collections::HashSet;

use im::HashMap;

use crate::constants::*;
use crate::error::*;

pub const ERROR_LABEL: &str = "throw_error";
pub const PRINT_LABEL: &str = "snek_print";
pub const STRUCT_EQ_LABEL: &str = "snek_struct_eq";

#[derive(Debug, Copy, Clone)]
enum Val {
    Reg(Reg),
    Imm(i64),
    StaticRegOffset(Reg, i64),
    DynamicRegOffset(Reg, Reg)
}



#[derive(Debug)]
enum VarTypes {
    NUM,
    BOOL,
}

#[derive(Debug, Copy, Clone)]
enum Reg {
    RAX,
    RSP,
    RDI,
    RBX,
    R15,
    RCX,
    RSI
}

use Reg::*;

#[derive(Debug)]
enum Instr {
    IMov(Val, Val),
    IAdd(Val, Val),
    ISub(Val, Val),
    IMul(Val, Val),
    ITest(Val, Val),
    ISar(Val, Val),
    ICmp(Val, Val),
    IJe(String),
    IJo(String),
    IJmp(String),
    IJnz(String),
    IJz(String),
    IJl(String),
    IJge(String),
    ILabel(String),
    ICmove(Val, Val),
    ICmovne(Val, Val),
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
    si: i64,
    env: &'a HashMap<String, i64>,
    brake: &'a str,
    defintions: &'a Vec<Definition>,
    func_mode: bool,
}

// Prompted Chatgpt to implement a builder class for the Context struct using the prompt:
// "Generate a builder class for the following struct <Insert My Context Implementation Here>"

struct ContextBuilder<'a> {
    si: i64,
    env: Option<&'a HashMap<String, i64>>,
    brake: Option<&'a str>,
    defintions: Option<&'a Vec<Definition>>,
    func_mode: bool,
}

impl<'a> ContextBuilder<'a> {
    fn new(
        si: i64,
        env: &'a HashMap<String, i64>,
        brake: &'a str,
        defintions: &'a Vec<Definition>,
        func_mode: bool,
    ) -> Self {
        Self {
            si,
            env: Some(env),
            brake: Some(brake),
            defintions: Some(defintions),
            func_mode,
        }
    }

    fn env(mut self, env: &'a HashMap<String, i64>) -> Self {
        self.env = Some(env);
        self
    }

    fn brake(mut self, brake: &'a str) -> Self {
        self.brake = Some(brake);
        self
    }

    fn si(mut self, si: i64) -> Self {
        self.si = si;
        self
    }

    fn from(context: &'a Context<'a>) -> Self {
        Self {
            si: context.si,
            env: Some(context.env),
            brake: Some(context.brake),
            defintions: Some(context.defintions),
            func_mode: context.func_mode,
        }
    }

    fn build(self) -> Context<'a> {
        Context {
            si: self.si,
            env: self.env.expect("env is not set"),
            brake: self.brake.expect("brake is not set"),
            defintions: self.defintions.expect("def not set"),
            func_mode: self.func_mode,
        }
    }
}

pub fn compile_program(program: &Program) -> (String, String) {
    let mut labels: i64 = 0;
    let mut defs: String = String::new();

    if duplicate_func_names(&program.defs) {
        panic!("Invalid duplicate func name")
    }
    for def in &program.defs[..] {
        defs.push_str(&compile_definition(&def, &mut labels, &program.defs));
    }
    let main = compile_expr_to_string(
        &program.main,
        &ContextBuilder::new(0, &HashMap::new(), &String::from(""), &program.defs, false).build(),
        &mut labels,
    );
    (defs, main)
}

fn compile_definition(d: &Definition, labels: &mut i64, definitions: &Vec<Definition>) -> String {
    match d {
        Definition::Func(funcname, args, body) => {
            let mut env = HashMap::new();
            let mut loc = 1;
            let mut depth = depth(body);
            depth = if depth % 2 == 0 { depth } else { depth + 1 };
            /*             println!("{:?}", d); */
            for arg in args {
                /*                 println!("arg: {} depth: {} loc: {}", arg, depth, loc);
                 */
                env.insert(arg.to_string(), 8 * depth + 8 * loc);
                loc += 1;
            }
            let body_instrs_as_str = compile_expr_to_string(
                body,
                &ContextBuilder::new(0, &env, &String::from(""), definitions, true).build(),
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
fn depth(e: &Expr) -> i64 {
    match e {
        Expr::Number(_) => 0,
        Expr::Boolean(_) => 0,
        Expr::UnOp(_, expr) => depth(expr),
        Expr::BinOp(_, e1, e2) => depth(e2).max(depth(e1) + 1), // For binops we compile RHS first for ease with Minus and Conditionals
        Expr::Let(bindings, body) => bindings
            .iter()
            .enumerate()
            .map(|(index, (_, expr))| depth(expr) + index as i64)
            .max()
            .unwrap_or(0)
            .max(depth(body) + bindings.len() as i64),
        Expr::Id(_) => 0,
        Expr::If(expr1, expr2, expr3) => depth(expr1).max(depth(expr2)).max(depth(expr3)),
        Expr::Loop(expr) => depth(expr),
        Expr::Block(exprs) => exprs.iter().map(|expr| depth(expr)).max().unwrap_or(0),
        Expr::Break(expr) => depth(expr),
        Expr::Set(_, expr) => depth(expr),
        Expr::Call(_, arg_exprs) => arg_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| depth(expr) + index as i64)
            .max()
            .unwrap_or(0),
        Expr::Tuple(elem_exprs) => elem_exprs
            .iter()
            .enumerate()
            .map(|(index, expr)| depth(expr) + index as i64)
            .max()
            .unwrap_or(0),
        Expr::TupleSet(tup, ind, val) => depth(val).max(depth(ind)+1).max(depth(tup)+2) // so that tup is in RAX
    }
}

fn compile_expr_to_string(e: &Expr, context: &Context, labels: &mut i64) -> String {
    let mut depth = depth(e);
    depth = if depth % 2 == 0 { depth } else { depth + 1 };
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

fn compile_to_instrs(expr: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i64) {
    match expr {
        Expr::Number(n) => instrs.push(Instr::IMov(Val::Reg(RAX), convert_i64_to_val(n))),
        Expr::Boolean(b) => instrs.push(Instr::IMov(Val::Reg(RAX), convert_bool_to_val(b))),
        Expr::Id(s) if s == "input" => {
            if !context.func_mode {
                instrs.push(Instr::IMov(Val::Reg(RAX), Val::Reg(RDI)))
            } else {
                panic!("Invalid function cannot use input")
            }
        }
        Expr::Id(s) if s == "nil" => {
            instrs.push(Instr::IMov(Val::Reg(RAX), Val::Imm(1)))
        }
        Expr::Id(id) => match context.env.contains_key(id) {
            true => instrs.push(Instr::IMov(
                Val::Reg(RAX),
                Val::StaticRegOffset(RSP, *context.env.get(id).unwrap()),
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
        Expr::Tuple(elem_exprs) => compile_tuple(elem_exprs, instrs, context, labels),
        Expr::TupleSet(tuple_expr, index_expr, new_val_expr) => compile_tuple_set(tuple_expr, index_expr, new_val_expr, instrs, context, labels),
    }
}

fn compile_tuple_set(tuple_expr: &Expr, index_expr: &Expr, new_val_expr: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i64) {
    compile_to_instrs(new_val_expr, instrs, context, labels);
    instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, context.si*8), Val::Reg(RAX)));

    compile_to_instrs(index_expr, instrs, &ContextBuilder::from(context).si(context.si+1).build(), labels);
    instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, 8+context.si*8), Val::Reg(RAX)));

    compile_to_instrs(tuple_expr, instrs, &ContextBuilder::from(context).si(context.si+2).build(), labels);

    assert_heap_address(instrs);

    instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8+context.si*8)));
    assert_num(instrs, Val::Reg(RBX));

    // OOB error for nil object
    instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OUT_OF_BOUNDS_ERROR_CODE)));
    instrs.push(Instr::IJe(ERROR_LABEL.to_string()));
    // OOB for less than 0 index
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8+context.si*8)));    
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(0)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OUT_OF_BOUNDS_ERROR_CODE)));
    instrs.push(Instr::IJl(ERROR_LABEL.to_string()));
    // OOB for greater than or equal to length index
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8+context.si*8)));
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::StaticRegOffset(RAX, -1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OUT_OF_BOUNDS_ERROR_CODE)));
    instrs.push(Instr::IJge(ERROR_LABEL.to_string()));
    // Move element from heap at index location into rax
    instrs.push(Instr::IMov(Val::Reg(RCX), Val::StaticRegOffset(RSP, 8*context.si))); // RCX holds new val
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8+context.si*8))); // RBX holds index
    instrs.push(Instr::ISar(Val::Reg(RBX), Val::Imm(1))); // Shift the index to the right to get the "correct index value"
    instrs.push(Instr::IAdd(Val::Reg(RBX), Val::Imm(1))); // Add in for metadata
    instrs.push(Instr::IMul(Val::Reg(RBX), Val::Imm(8))); // Calculate actual heap offset
    instrs.push(Instr::ISub(Val::Reg(RAX), Val::Imm(1))); // temporary mangle due to type
    instrs.push(Instr::IMov(Val::DynamicRegOffset(RAX, RBX), Val::Reg(RCX)));
    instrs.push(Instr::IAdd(Val::Reg(RAX), Val::Imm(1))); // Add back 1 to insert the type for the heap address
}

fn compile_tuple(
    elem_exprs: &Vec<Expr>,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i64,
) {

    // Calculate elements and store in stack (last element in RAX)
    let mut total_stack_offset = context.si*8;
    for (index, elem_expr) in elem_exprs.iter().take(elem_exprs.len()-1).enumerate() {
        compile_to_instrs(elem_expr, instrs, &ContextBuilder::from(context).si(index as i64 + context.si).build(), labels);
        instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, total_stack_offset), Val::Reg(RAX)));
        total_stack_offset += 8;
    }
    compile_to_instrs(
        elem_exprs.get(elem_exprs.len()-1).unwrap(), 
        instrs, &ContextBuilder::from(context).si(elem_exprs.len() as i64 + context.si-1).build(), 
        labels);
    // Store length of tuple at R15
    instrs.push(Instr::IMov(Val::Reg(RBX), convert_i64_to_val(&(elem_exprs.len() as i64))));
    instrs.push(Instr::IMov(Val::StaticRegOffset(R15, 0), Val::Reg(RBX)));
    // Move Elements into Heap
    let mut stack_offset = context.si*8;
    let mut heap_offset = 8; // Starting at 8 since length is "first element"

    instrs.push(Instr::IMov(Val::StaticRegOffset(R15, (elem_exprs.len() as i64) * 8), Val::Reg(RAX))); // for the last element

    while stack_offset < total_stack_offset {
        instrs.push(Instr::IMov(Val::Reg(RAX), Val::StaticRegOffset(RSP, stack_offset)));
        instrs.push(Instr::IMov(Val::StaticRegOffset(R15, heap_offset), Val::Reg(RAX)));
        stack_offset += 8;
        heap_offset += 8;
    }

    // Align offset for next tuple
    if heap_offset % 16 != 0 {
        heap_offset += 8; // Either going to be 0 or 8
    }
    // Update RAX
    instrs.push(Instr::IMov(Val::Reg(RAX), Val::Reg(R15)));
    instrs.push(Instr::IAdd(Val::Reg(RAX), Val::Imm(1)));
    // Update R15
    if elem_exprs.len() % 2 == 0 {
        heap_offset += 8;
    }
    instrs.push(Instr::IAdd(Val::Reg(R15), Val::Imm(heap_offset.into())));
}

fn compile_func_call(
    funcname: &String,
    arg_exprs: &Vec<Expr>,
    instrs: &mut Vec<Instr>,
    context: &Context,
    labels: &mut i64,
) {
    let mut found = false;
    let mut arg_count = 0;
    for definition in context.defintions {
        match definition {
            Definition::Func(name, args, _) => {
                if name == funcname {
                    found = true;
                    arg_count = args.len();
                }
            }
        }
    }
    if !found {
        panic!("Invalid func doesn't exist: {}", funcname)
    }

    if arg_exprs.len() != arg_count {
        panic!("Invalid number of args is incorrect for funcname: {}", funcname)
    }

    let mut offset = (arg_exprs.len() as i64 + 1) * 8;
    if offset % 16 == 0 {
        offset += 8;
    }
    let mut cur_word = context.si * 8;
    let mut new_si = context.si;
    let mut curr_word_after_sub = offset + cur_word;
    if arg_exprs.len() > 0 {
        for arg_expr in arg_exprs.iter().take(arg_exprs.len() - 1) {
            compile_to_instrs(
                arg_expr,
                instrs,
                &ContextBuilder::from(context).si(new_si).build(),
                labels,
            );
            instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, cur_word), Val::Reg(RAX)));
            cur_word += 8;
            new_si = new_si + 1;
        }
        compile_to_instrs(
            arg_exprs.get(arg_exprs.len() - 1).unwrap(),
            instrs,
            &ContextBuilder::from(context).si(new_si).build(),
            labels,
        );
    }
    instrs.push(Instr::ISub(Val::Reg(RSP), Val::Imm(offset.into())));
    // Move args in stack to be near new RSP location
    let mut cur_offset_to_new_loc = 0;
    if arg_exprs.len() > 0 {
        for _ in arg_exprs.iter().take(arg_exprs.len() - 1) {
            instrs.push(Instr::IMov(
                Val::Reg(RBX),
                Val::StaticRegOffset(RSP, curr_word_after_sub),
            ));
            instrs.push(Instr::IMov(
                Val::StaticRegOffset(RSP, cur_offset_to_new_loc),
                Val::Reg(RBX),
            ));
            curr_word_after_sub += 8;
            cur_offset_to_new_loc += 8;
        }
        instrs.push(Instr::IMov(
            Val::StaticRegOffset(RSP, cur_offset_to_new_loc),
            Val::Reg(RAX),
        ));
        cur_offset_to_new_loc += 8;
    }
    instrs.push(Instr::IMov(
        Val::StaticRegOffset(RSP, cur_offset_to_new_loc),
        Val::Reg(RDI),
    ));
    // Call
    instrs.push(Instr::ICall(funcname.to_string()));
    // Restore rdi
    instrs.push(Instr::IMov(
        Val::Reg(RDI),
        Val::StaticRegOffset(RSP, cur_offset_to_new_loc),
    ));
    // Restore stack pointer
    instrs.push(Instr::IAdd(Val::Reg(RSP), Val::Imm(offset.into())));
}

fn compile_break(e: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i64) {
    if context.brake == "" {
        panic!("Error: break occurred outside a loop in sexp")
    }
    compile_to_instrs(e, instrs, context, labels);
    instrs.push(Instr::IJmp(context.brake.to_string()));
}

fn compile_loop(e: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i64) {
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
    labels: &mut i64,
) {
    let end_label = new_label(labels, "ifend");
    let else_label = new_label(labels, "ifelse");

    compile_to_instrs(cond, instrs, context, labels);
    instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(3)));
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
    labels: &mut i64,
) {
    if !context.env.contains_key(id) {
        panic!("Unbound variable identifier {id}")
    } else {
        compile_to_instrs(e, instrs, context, labels);
        instrs.push(Instr::IMov(
            Val::StaticRegOffset(RSP, *context.env.get(id).unwrap()),
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
    labels: &mut i64,
) {
    compile_to_instrs(e2, instrs, context, labels);
    instrs.push(Instr::IMov(
        Val::StaticRegOffset(RSP, 8 * context.si),
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
            assert_both_num(instrs, Val::StaticRegOffset(RSP, 8 * context.si), Val::Reg(RAX));
            instrs.push(Instr::IAdd(
                Val::Reg(RAX),
                Val::StaticRegOffset(RSP, 8 * context.si),
            ));
            check_for_overflow(instrs);
        }
        Op2::Minus => {
            assert_both_num(instrs, Val::StaticRegOffset(RSP, 8 * context.si), Val::Reg(RAX));
            instrs.push(Instr::ISub(
                Val::Reg(RAX),
                Val::StaticRegOffset(RSP, 8 * context.si),
            ));
            check_for_overflow(instrs);
        }
        Op2::Times => {
            assert_both_num(instrs, Val::StaticRegOffset(RSP, 8 * context.si), Val::Reg(RAX));
            instrs.push(Instr::ISar(Val::Reg(RAX), Val::Imm(1)));
            instrs.push(Instr::IMul(
                Val::Reg(RAX),
                Val::StaticRegOffset(RSP, 8 * context.si),
            ));
            check_for_overflow(instrs);
        }
        Op2::TupleGet => {
            /* tuple is in RAX, index is in RSP + 8*context.si */

            // Check if RAX hold a heap address
            assert_heap_address(instrs);
            // Copy index from RSP + 8*context.si into RBX and check if it is a number
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8*context.si)));
            assert_num(instrs, Val::Reg(RBX));
            // Check if it is nil and throw out of bounds
            instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(1)));
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OUT_OF_BOUNDS_ERROR_CODE)));
            instrs.push(Instr::IJe(ERROR_LABEL.to_string()));
            // Throw runtime error if index is less than 0
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8*context.si)));
            instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(0)));
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OUT_OF_BOUNDS_ERROR_CODE)));
            instrs.push(Instr::IJl(ERROR_LABEL.to_string()));
            // Throw runtime error if index is greater than or equal to length of tuple
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8*context.si)));
            instrs.push(Instr::ICmp(Val::Reg(RBX), Val::StaticRegOffset(RAX, -1)));
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OUT_OF_BOUNDS_ERROR_CODE)));
            instrs.push(Instr::IJge(ERROR_LABEL.to_string()));
            // Move element from heap at index location into rax
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::StaticRegOffset(RSP, 8*context.si)));
            instrs.push(Instr::ISar(Val::Reg(RBX), Val::Imm(1))); // Shift the index to the right to get the "correct index value"
            instrs.push(Instr::IAdd(Val::Reg(RBX), Val::Imm(1))); // Add in for metadata
            instrs.push(Instr::IMul(Val::Reg(RBX), Val::Imm(8))); // Calculate actual heap offset
            instrs.push(Instr::ISub(Val::Reg(RAX), Val::Imm(1))); // mangle due to type
            instrs.push(Instr::IMov(Val::Reg(RAX), Val::DynamicRegOffset(RAX, RBX)));
        }
        Op2::StructEqual => {
            /* arg1 is in RAX, arg2 is in RSP + 8*context.si */
            instrs.push(Instr::IMov(Val::Reg(RSI), Val::StaticRegOffset(RSP, 8*context.si)));
            let offset = 8 * (context.si + 1 + (context.si % 2));
            instrs.push(Instr::ISub(Val::Reg(RSP), Val::Imm(offset.into())));
            instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, 0), Val::Reg(RDI)));
            instrs.push(Instr::IMov(Val::Reg(RDI), Val::Reg(RAX)));
            instrs.push(Instr::ICall(STRUCT_EQ_LABEL.to_string()));
            instrs.push(Instr::IMov(Val::Reg(RDI), Val::StaticRegOffset(RSP, 0)));
            instrs.push(Instr::IAdd(Val::Reg(RSP), Val::Imm(offset.into())));
        }
        _ => compile_conditional_binop(op, instrs, context, labels),
    }
}

fn compile_conditional_binop(op: &Op2, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i64) {
    match op {
        Op2::Equal => {
            assert_both_same_type(instrs, Val::Reg(RAX), Val::StaticRegOffset(RSP, 8 * context.si), labels)
        }
        _ => assert_both_num(instrs, Val::Reg(RAX), Val::StaticRegOffset(RSP, 8 * context.si)),
    };
    instrs.push(Instr::ICmp(
        Val::Reg(RAX),
        Val::StaticRegOffset(RSP, 8 * context.si),
    ));
    instrs.push(Instr::IMov(Val::Reg(RAX), Val::Imm(3)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(7)));
    match op {
        Op2::Equal => instrs.push(Instr::ICmove(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::Greater => instrs.push(Instr::ICmovg(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::GreaterEqual => instrs.push(Instr::ICmovge(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::Less => instrs.push(Instr::ICmovl(Val::Reg(RAX), Val::Reg(RBX))),
        Op2::LessEqual => instrs.push(Instr::ICmovle(Val::Reg(RAX), Val::Reg(RBX))),
        _ => unreachable!(),
    }
}

fn compile_unop(op: &Op1, e: &Expr, instrs: &mut Vec<Instr>, context: &Context, labels: &mut i64) {
    compile_to_instrs(e, instrs, context, labels);
    match op {
        Op1::Add1 => {
            assert_num(instrs, Val::Reg(RAX));
            instrs.push(Instr::IAdd(Val::Reg(RAX), Val::Imm(2)));
            check_for_overflow(instrs);
        }
        Op1::Sub1 => {
            assert_num(instrs, Val::Reg(RAX));
            instrs.push(Instr::ISub(Val::Reg(RAX), Val::Imm(2)));
            check_for_overflow(instrs);
        }
        Op1::IsNum => check_type(instrs, VarTypes::NUM),
        Op1::IsBool => check_type(instrs, VarTypes::BOOL),
        Op1::Print => {
            let offset = 8 * (context.si + 1 + (context.si % 2));
            instrs.push(Instr::ISub(Val::Reg(RSP), Val::Imm(offset.into())));
            instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, 0), Val::Reg(RDI)));
            instrs.push(Instr::IMov(Val::Reg(RDI), Val::Reg(RAX)));
            instrs.push(Instr::ICall(PRINT_LABEL.to_string()));
            instrs.push(Instr::IMov(Val::Reg(RDI), Val::StaticRegOffset(RSP, 0)));
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
    labels: &mut i64,
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
        instrs.push(Instr::IMov(Val::StaticRegOffset(RSP, 8 * new_si), Val::Reg(RAX)));
        new_env = new_env.update(identifier.to_string(), new_si * 8);
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
        true => Val::Imm(7),
        false => Val::Imm(3),
    }
}

fn new_label(l: &mut i64, s: &str) -> String {
    let current = *l;
    *l += 1;
    format!("{s}_{current}")
}

fn assert_heap_address(instrs: &mut Vec<Instr>) {
    // Check that it isn't an integer
    instrs.push(Instr::ITest(Val::Reg(RAX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJz(ERROR_LABEL.to_string()));
    // Check that it isn't 3 (false)
    instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(3)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJe(ERROR_LABEL.to_string()));
    // Check that it isn't 7 (true)
    instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(7)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJe(ERROR_LABEL.to_string()));
}

fn assert_num(instrs: &mut Vec<Instr>, v: Val) {
    instrs.push(Instr::ITest(v, Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJnz(ERROR_LABEL.to_string()));
}

fn assert_both_num(instrs: &mut Vec<Instr>, r1: Val, r2: Val) {
    instrs.push(Instr::IMov(Val::Reg(RBX), r1));
    instrs.push(Instr::IOr(Val::Reg(RBX), r2));
    instrs.push(Instr::IAnd(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::ITest(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJnz(ERROR_LABEL.to_string()));
}

fn assert_both_same_type(instrs: &mut Vec<Instr>, r1: Val, r2: Val, labels: &mut i64) {
    let check_num_label = new_label(labels, "check_num");
    let check_bool_label = new_label(labels, "check_bool");
    let end_check_type_label = new_label(labels, "end_check_type");
    // Check if last bit is 0 and jumps to number checker label
    instrs.push(Instr::IMov(Val::Reg(RBX), r1));
    instrs.push(Instr::ITest(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::IJz(check_num_label.to_string()));

    // Checks if first arg is a boolean and jumps to a boolean checker label
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(3))); // false
    instrs.push(Instr::IJe(check_bool_label.to_string()));
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(7))); // true
    instrs.push(Instr::IJe(check_bool_label.to_string()));

    // At this point just check that the second arg ends with 1 and isn't 3 or 7
    instrs.push(Instr::IMov(Val::Reg(RBX), r2));
    instrs.push(Instr::ITest(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJz(ERROR_LABEL.to_string()));
    instrs.push(Instr::IMov(Val::Reg(RBX), r2));
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(3))); // false
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJe(ERROR_LABEL.to_string()));
    instrs.push(Instr::IMov(Val::Reg(RBX), r2));
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(7))); // true
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJe(ERROR_LABEL.to_string()));
    instrs.push(Instr::IJmp(end_check_type_label.to_string()));

    // boolean checker label - Checks that second arg is a bool
    instrs.push(Instr::ILabel(check_bool_label.to_string()));
    instrs.push(Instr::IMov(Val::Reg(RBX), r2));
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(3))); // false
    instrs.push(Instr::IJe(end_check_type_label.to_string()));
    instrs.push(Instr::ICmp(Val::Reg(RBX), Val::Imm(7))); // true
    instrs.push(Instr::IJe(end_check_type_label.to_string()));

    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJmp(ERROR_LABEL.to_string()));

    // number checker label - Checks that second arg is a num
    instrs.push(Instr::ILabel(check_num_label.to_string()));
    instrs.push(Instr::IMov(Val::Reg(RBX), r2));
    instrs.push(Instr::ITest(Val::Reg(RBX), Val::Imm(1)));
    instrs.push(Instr::IJz(end_check_type_label.to_string()));

    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(INVALID_ARGUMENT_CODE)));
    instrs.push(Instr::IJmp(ERROR_LABEL.to_string()));

    // End label for all type checks if type check was successful
    instrs.push(Instr::ILabel(end_check_type_label.to_string()));
}

fn check_for_overflow(instrs: &mut Vec<Instr>) {
    instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(OVERFLOW_ERROR_CODE)));
    instrs.push(Instr::IJo(ERROR_LABEL.to_string()));
}

fn check_type(instrs: &mut Vec<Instr>, type_to_check: VarTypes) {
    // TODO: Fix for Tuple
    match type_to_check {
        VarTypes::NUM => {
            instrs.push(Instr::ITest(Val::Reg(RAX), Val::Imm(1)));
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(7)));
            instrs.push(Instr::IMov(Val::Reg(RAX), Val::Imm(3)));
            instrs.push(Instr::ICmovz(Val::Reg(RAX), Val::Reg(RBX)));
        }
        VarTypes::BOOL => {
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(7)));
            instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(7)));
            instrs.push(Instr::ICmove(Val::Reg(RAX), Val::Reg(RBX)));
            instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(3)));
            instrs.push(Instr::ICmove(Val::Reg(RAX), Val::Reg(RBX)));

            instrs.push(Instr::ICmp(Val::Reg(RAX), Val::Imm(7)));
            instrs.push(Instr::IMov(Val::Reg(RBX), Val::Imm(3)));
            instrs.push(Instr::ICmovne(Val::Reg(RAX), Val::Reg(RBX)));
        }
    }
}
fn instr_to_str(i: &Instr) -> String {
    match i {
        Instr::IMov(v1, v2) => format!("\nmov {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IAdd(v1, v2) => format!("\nadd {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IMul(v1, v2) => format!("\nimul {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ISub(v1, v2) => format!("\nsub {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ITest(v1, v2) => format!("\ntest {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IOr(v1, v2) => format!("\nor {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IAnd(v1, v2) => format!("\nand {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmp(v1, v2) => format!("\ncmp {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::IJe(s) => format!("\nje {}", s.to_string()),
        Instr::IJmp(s) => format!("\njmp {}", s.to_string()),
        Instr::ILabel(s) => format!("\n {}:", s.to_string()),
        Instr::IJo(s) => format!("\njo {}", s.to_string()),
        Instr::IJz(s) => format!("\njz {}", s.to_string()),
        Instr::IJnz(s) => format!("\njnz {}", s.to_string()),
        Instr::IJl(s) => format!("\njl {}", s.to_string()),
        Instr::IJge(s) => format!("\njge {}", s.to_string()),
        Instr::ISar(v1, v2) => format!("\nsar {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmove(v1, v2) => format!("\ncmove {}, {}", val_to_str(v1), val_to_str(v2)),
        Instr::ICmovne(v1, v2) => format!("\ncmovne {}, {}", val_to_str(v1), val_to_str(v2)),
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
        Val::StaticRegOffset(r, n) => format!("[{}+{}]", reg_to_str(r), n),
        Val::DynamicRegOffset(r1, r2) => format!("[{}+{}]", reg_to_str(r1), reg_to_str(r2)),
    }
}

fn reg_to_str(r: &Reg) -> String {
    match r {
        RAX => "rax".to_string(),
        RSP => "rsp".to_string(),
        RDI => "rdi".to_string(),
        RBX => "rbx".to_string(),
        R15 => "r15".to_string(),
        RCX => "rcx".to_string(),
        RSI => "rsi".to_string()
    }
}
