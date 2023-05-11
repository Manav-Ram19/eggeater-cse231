use std::env;


#[link(name = "our_code")]
extern "C" {
    // The \x01 here is an undocumented feature of LLVM that ensures
    // it does not add an underscore in front of the name.
    // Courtesy of Max New (https://maxsnew.com/teaching/eecs-483-fa22/hw_adder_assignment.html)
    #[link_name = "\x01our_code_starts_here"]
    fn our_code_starts_here(input: u64) -> u64;
}

#[export_name = "\x01snek_error"]
pub extern "C" fn snek_error(errcode: i64) {
    match errcode {
        1 => eprintln!("overflow occured with error code: {errcode}"),
        2 => eprintln!("invalid argument occured with error code: {errcode}"),
        _ => eprintln!("an error ocurred {errcode}")
    }
    std::process::exit(1);
}

fn parse_input(input: &str) -> u64 {
    match input {
        "true" => 3,
        "false" => 1,
        _ => match input.parse::<i64>() {
            Ok(n) => {
                if n > 4611686018427387903 || n < -4611686018427387904 {
                    panic!("Invalid out of bounds input: {}", input)
                } else {
                    return (n << 1) as u64;
                }
            }
            _ => {
                panic!("Invalid input: {}", input)
            }
        }
    }
}

fn print_value(val : i64) {
    if val == 3 { println!("true"); }
    else if val == 1 { println!("false"); }
    else if val % 2 == 0 { println!("{}", val >> 1); }
    else {
      println!("Unknown value: {}", val);
    }
  }

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = if args.len() == 2 { &args[1] } else { "false" };
    let input = parse_input(&input);

    let i: u64 = unsafe { our_code_starts_here(input) };
    print_value(i as i64);
}