package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.distributed.*;

/**
 * Pickle protocol opcodes.
 * 
 * <p>Reference: Python's pickle.py and pickletools.py documentation.</p>
 * 
 * <p>Protocol versions:
 * <ul>
 *   <li>0, 1: text-based opcodes</li>
 *   <li>2: binary integers, NEWOBJ, TUPLE1/2/3</li>
 *   <li>3: binary bytes (Python 3)</li>
 *   <li>4: frame, BINUNICODE8, etc (Python 3.4+)</li>
 *   <li>5: out-of-band buffers (Python 3.8+)</li>
 * </ul>
 * </p>
 */
public interface PickleOpcodes {
    
    // === Protocol 0 & 1 opcodes (text-based) ===
    
    /** Push special markobject on stack */
    short MARK = '(';
    
    /** Every pickle ends with STOP */
    short STOP = '.';
    
    /** Discard topmost stack item */
    short POP = '0';
    
    /** Discard stack top through topmost markobject */
    short POP_MARK = '1';
    
    /** Duplicate top stack item */
    short DUP = '2';
    
    /** Push float object; decimal string argument */
    short FLOAT = 'F';
    
    /** Push integer or bool; decimal string argument */
    short INT = 'I';
    
    /** Push four-byte signed int (little endian) */
    short BININT = 'J';
    
    /** Push 1-byte unsigned int */
    short BININT1 = 'K';
    
    /** Push long; decimal string argument */
    short LONG = 'L';
    
    /** Push 2-byte unsigned int */
    short BININT2 = 'M';
    
    /** Push None */
    short NONE = 'N';
    
    /** Push persistent object; id is taken from string arg */
    short PERSID = 'P';
    
    /** Push persistent object; id is taken from stack */
    short BINPERSID = 'Q';
    
    /** Apply callable to argtuple, both on stack */
    short REDUCE = 'R';
    
    /** Push string; NL-terminated string argument */
    short STRING = 'S';
    
    /** Push string; counted binary string argument */
    short BINSTRING = 'T';
    
    /** Push string; counted binary string < 256 bytes */
    short SHORT_BINSTRING = 'U';
    
    /** Push Unicode string; raw-unicode-escaped'd argument */
    short UNICODE = 'V';
    
    /** Push Unicode string; counted UTF-8 string argument */
    short BINUNICODE = 'X';
    
    /** Append stack top to list below it */
    short APPEND = 'a';
    
    /** Call __setstate__ or __dict__.update() */
    short BUILD = 'b';
    
    /** Push self.find_class(modname, name); 2 string args */
    short GLOBAL = 'c';
    
    /** Build a dict from stack items */
    short DICT = 'd';
    
    /** Push empty dict */
    short EMPTY_DICT = '}';
    
    /** Extend list on stack by topmost stack slice */
    short APPENDS = 'e';
    
    /** Push item from memo on stack; index is string arg */
    short GET = 'g';
    
    /** Push item from memo on stack; index is 1-byte arg */
    short BINGET = 'h';
    
    /** Build & push class instance */
    short INST = 'i';
    
    /** Push item from memo on stack; index is 4-byte arg */
    short LONG_BINGET = 'j';
    
    /** Build list from topmost stack items */
    short LIST = 'l';
    
    /** Push empty list */
    short EMPTY_LIST = ']';
    
    /** Build & push class instance */
    short OBJ = 'o';
    
    /** Store stack top in memo; index is string arg */
    short PUT = 'p';
    
    /** Store stack top in memo; index is 1-byte arg */
    short BINPUT = 'q';
    
    /** Store stack top in memo; index is 4-byte arg */
    short LONG_BINPUT = 'r';
    
    /** Add key+value pair to dict */
    short SETITEM = 's';
    
    /** Build tuple from topmost stack items */
    short TUPLE = 't';
    
    /** Push empty tuple */
    short EMPTY_TUPLE = ')';
    
    /** Modify dict by adding topmost key+value pairs */
    short SETITEMS = 'u';
    
    /** Push float; arg is 8-byte big-endian float encoding */
    short BINFLOAT = 'G';
    
    // === Protocol 2 opcodes (binary integers) ===
    
    /** Identify pickle protocol */
    short PROTO = 0x80;
    
    /** Build object by applying cls.__new__ to argtuple */
    short NEWOBJ = 0x81;
    
    /** Push object from extension registry; 1-byte index */
    short EXT1 = 0x82;
    
    /** Push object from extension registry; 2-byte index */
    short EXT2 = 0x83;
    
    /** Push object from extension registry; 4-byte index */
    short EXT4 = 0x84;
    
    /** Build 1-tuple from stack top */
    short TUPLE1 = 0x85;
    
    /** Build 2-tuple from two topmost stack items */
    short TUPLE2 = 0x86;
    
    /** Build 3-tuple from three topmost stack items */
    short TUPLE3 = 0x87;
    
    /** Push True */
    short NEWTRUE = 0x88;
    
    /** Push False */
    short NEWFALSE = 0x89;
    
    /** Push long from < 256 bytes */
    short LONG1 = 0x8a;
    
    /** Push really big long */
    short LONG4 = 0x8b;
    
    // === Protocol 3 opcodes (Python 3.x binary bytes) ===
    
    /** Push bytes; counted binary string argument */
    short BINBYTES = 'B';
    
    /** Push bytes; counted binary string < 256 bytes */
    short SHORT_BINBYTES = 'C';
    
    // === Protocol 4 opcodes (Python 3.4+) ===
    
    /** Push short string; UTF-8 length < 256 bytes */
    short SHORT_BINUNICODE = 0x8c;
    
    /** Push very long string */
    short BINUNICODE8 = 0x8d;
    
    /** Push very long bytes string */
    short BINBYTES8 = 0x8e;
    
    /** Push empty set on the stack */
    short EMPTY_SET = 0x8f;
    
    /** Modify set by adding topmost stack items */
    short ADDITEMS = 0x90;
    
    /** Build frozenset from topmost stack items */
    short FROZENSET = 0x91;
    
    /** Like NEWOBJ but work with keyword only arguments */
    short NEWOBJ_EX = 0x92;
    
    /** Same as GLOBAL but using names on the stacks */
    short STACK_GLOBAL = 0x93;
    
    /** Store top of the stack in memo */
    short MEMOIZE = 0x94;
    
    /** Indicate the beginning of a new frame */
    short FRAME = 0x95;
    
    // === Protocol 5 opcodes (Python 3.8+) ===
    
    /** Push bytearray */
    short BYTEARRAY8 = 0x96;
    
    /** Push next out-of-band buffer */
    short NEXT_BUFFER = 0x97;
    
    /** Make top of stack readonly */
    short READONLY_BUFFER = 0x98;
}
