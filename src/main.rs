use std::env;
use std::fs::File;
use std::io::prelude::*;

use diamondback::compiler::compile_program;
use diamondback::parser::parse_program;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();

    let in_name = &args[1];
    let out_name = &args[2];

    let mut in_file = File::open(in_name)?;
    let mut in_contents = String::new();
    in_file.read_to_string(&mut in_contents)?;

    let parsed_program = parse_program(&in_contents);
    let (defs, main) = compile_program(&parsed_program);

    let asm_program = format!(
      "
      section .text
      global our_code_starts_here
      extern snek_error
      extern snek_print
      throw_error:
      mov rdi, rbx
      push rsp
      call snek_error
      {}
      our_code_starts_here:
        {}
        ret
      ",
        defs,
        main
    );
/*     print!("{}", asm_program); */
    let mut out_file = File::create(out_name)?;
    out_file.write_all(asm_program.as_bytes())?;

    Ok(())
}
