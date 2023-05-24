use std::env;


#[link(name = "our_code")]
extern "C" {
    // The \x01 here is an undocumented feature of LLVM that ensures
    // it does not add an underscore in front of the name.
    // Courtesy of Max New (https://maxsnew.com/teaching/eecs-483-fa22/hw_adder_assignment.html)
    #[link_name = "\x01our_code_starts_here"]
    fn our_code_starts_here(input : i64, memory : *mut i64) -> i64;
}

#[export_name = "\x01snek_error"]
pub extern "C" fn snek_error(errcode: i64) {
    match errcode {
        1 => eprintln!("overflow occured with error code: {errcode}"),
        2 => eprintln!("invalid argument occured with error code: {errcode}"),
        3 => eprintln!("out of bounds error occured with error code: {errcode}"),
        _ => eprintln!("an error ocurred {errcode}")
    }
    std::process::exit(1);
}

fn snek_str(val : i64, seen : &mut Vec<i64>) -> String {
    if val == 7 { "true".to_string() }
    else if val == 3 { "false".to_string() }
    else if val % 2 == 0 { format!("{}", val >> 1) }
    else if val == 1 { "nil".to_string() }
    else if val & 1 == 1 {
      if seen.contains(&val)  { return "(tuple <cyclic>)".to_string() }
      seen.push(val);
      let addr = (val - 1) as *const i64;
      let length = (unsafe { *addr }) >> 1;
      let mut i = 1;
      let mut result = format!("(tuple");
      while i <= length {
        let elem = unsafe { *addr.offset(i as isize) };
        result = result + &format!(" {}", snek_str(elem, seen));
        i = i + 1; 
      }
      result = result + &format!(")");
      return result;
    }
    else {
      format!("Unknown value: {}", val)
    }
  }

#[no_mangle]
#[export_name = "\x01snek_print"]
fn snek_print(val : i64) -> i64 {
    let mut seen = Vec::<i64>::new();
    println!("{}", snek_str(val, &mut seen));
    return val;
}

fn parse_input(input: &str) -> i64 {
    match input {
        "true" => 7,
        "false" => 3,
        _ => match input.parse::<i64>() {
            Ok(n) => {
                if n > 4611686018427387903 || n < -4611686018427387904 {
                    panic!("Invalid out of bounds input: {}", input)
                } else {
                    return (n << 1) as i64;
                }
            }
            _ => {
                panic!("Invalid input: {}", input)
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let input = if args.len() == 2 { &args[1] } else { "false" };
    let input = parse_input(&input);

    let mut memory = Vec::<i64>::with_capacity(1000000);
    let buffer :*mut i64 = memory.as_mut_ptr();

    let i: i64 = unsafe { our_code_starts_here(input, buffer) };
    snek_print(i);
}