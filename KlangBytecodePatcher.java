import org.objectweb.asm.*;
import org.objectweb.asm.tree.*;
import org.objectweb.asm.util.Printer;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.jar.*;
import java.util.zip.*;

/** Safe, explicit ASM patcher used by Klang Discord bot. */
public final class KlangBytecodePatcher implements Opcodes {
  private static String valueOf(AbstractInsnNode n) {
    if (n instanceof LdcInsnNode) return String.valueOf(((LdcInsnNode)n).cst);
    if (n instanceof IntInsnNode) return Integer.toString(((IntInsnNode)n).operand);
    if (n instanceof InsnNode) {
      int op=n.getOpcode();
      if(op>=ICONST_M1 && op<=ICONST_5) return Integer.toString(op-ICONST_0);
      if(op==LCONST_0||op==LCONST_1) return Long.toString(op-LCONST_0);
      if(op==FCONST_0||op==FCONST_1||op==FCONST_2) return Float.toString(op-FCONST_0);
      if(op==DCONST_0||op==DCONST_1) return Double.toString(op-DCONST_0);
    }
    return null;
  }
  private static Object parse(String type,String value) {
    switch(type.toLowerCase(Locale.ROOT)) {
      case "string": return value;
      case "int": return Integer.valueOf(value);
      case "long": return Long.valueOf(value);
      case "float": return Float.valueOf(value);
      case "double": return Double.valueOf(value);
      default: throw new IllegalArgumentException("Unsupported constant type: "+type);
    }
  }
  private static AbstractInsnNode instruction(Object v) {
    if(v instanceof Integer) {
      int x=(Integer)v;
      if(x>=-1&&x<=5) return new InsnNode(ICONST_0+x);
      if(x>=Byte.MIN_VALUE&&x<=Byte.MAX_VALUE) return new IntInsnNode(BIPUSH,x);
      if(x>=Short.MIN_VALUE&&x<=Short.MAX_VALUE) return new IntInsnNode(SIPUSH,x);
    }
    if(v instanceof Long) { long x=(Long)v; if(x==0||x==1) return new InsnNode(x==0?LCONST_0:LCONST_1); }
    if(v instanceof Float) { float x=(Float)v; if(x==0||x==1||x==2) return new InsnNode(x==0?FCONST_0:x==1?FCONST_1:FCONST_2); }
    if(v instanceof Double) { double x=(Double)v; if(x==0||x==1) return new InsnNode(x==0?DCONST_0:DCONST_1); }
    return new LdcInsnNode(v);
  }
  private static boolean matchesType(AbstractInsnNode n,String type) {
    String t=type.toLowerCase(Locale.ROOT);
    if(t.equals("string")) return n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof String;
    if(t.equals("int")) return n instanceof IntInsnNode || (n instanceof InsnNode && n.getOpcode()>=ICONST_M1 && n.getOpcode()<=ICONST_5) || (n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Integer);
    if(t.equals("long")) return (n instanceof InsnNode && (n.getOpcode()==LCONST_0||n.getOpcode()==LCONST_1)) || (n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Long);
    if(t.equals("float")) return (n instanceof InsnNode && n.getOpcode()>=FCONST_0 && n.getOpcode()<=FCONST_2) || (n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Float);
    if(t.equals("double")) return (n instanceof InsnNode && (n.getOpcode()==DCONST_0||n.getOpcode()==DCONST_1)) || (n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Double);
    return false;
  }
  private static final Map<String,Integer> SAFE_OPCODES = new LinkedHashMap<>();
  static {
    SAFE_OPCODES.put("IFEQ", IFEQ); SAFE_OPCODES.put("IFNE", IFNE);
    SAFE_OPCODES.put("IFLT", IFLT); SAFE_OPCODES.put("IFGE", IFGE);
    SAFE_OPCODES.put("IFGT", IFGT); SAFE_OPCODES.put("IFLE", IFLE);
    SAFE_OPCODES.put("IF_ICMPEQ", IF_ICMPEQ); SAFE_OPCODES.put("IF_ICMPNE", IF_ICMPNE);
    SAFE_OPCODES.put("IF_ICMPLT", IF_ICMPLT); SAFE_OPCODES.put("IF_ICMPGE", IF_ICMPGE);
    SAFE_OPCODES.put("IF_ICMPGT", IF_ICMPGT); SAFE_OPCODES.put("IF_ICMPLE", IF_ICMPLE);
    SAFE_OPCODES.put("IF_ACMPEQ", IF_ACMPEQ); SAFE_OPCODES.put("IF_ACMPNE", IF_ACMPNE);
    SAFE_OPCODES.put("IFNULL", IFNULL); SAFE_OPCODES.put("IFNONNULL", IFNONNULL);
  }
  private static String opcodeName(int opcode) {
    for (Map.Entry<String,Integer> e : SAFE_OPCODES.entrySet())
      if (e.getValue() == opcode) return e.getKey();
    return opcode >= 0 && opcode < Printer.OPCODES.length && Printer.OPCODES[opcode] != null
      ? Printer.OPCODES[opcode] : "OP_" + opcode;
  }
  private static boolean safeOpcodePair(String oldName, String newName) {
    Set<Set<String>> pairs = new HashSet<>();
    pairs.add(new HashSet<>(Arrays.asList("IFEQ","IFNE")));
    pairs.add(new HashSet<>(Arrays.asList("IFLT","IFGE")));
    pairs.add(new HashSet<>(Arrays.asList("IFGT","IFLE")));
    pairs.add(new HashSet<>(Arrays.asList("IF_ICMPEQ","IF_ICMPNE")));
    pairs.add(new HashSet<>(Arrays.asList("IF_ICMPLT","IF_ICMPGE")));
    pairs.add(new HashSet<>(Arrays.asList("IF_ICMPGT","IF_ICMPLE")));
    pairs.add(new HashSet<>(Arrays.asList("IF_ACMPEQ","IF_ACMPNE")));
    pairs.add(new HashSet<>(Arrays.asList("IFNULL","IFNONNULL")));
    return !oldName.equals(newName) && pairs.contains(new HashSet<>(Arrays.asList(oldName,newName)));
  }
  private static void inspect(String jar,String out) throws Exception {
    try(JarFile jf=new JarFile(jar); PrintWriter w=new PrintWriter(Files.newBufferedWriter(Paths.get(out),StandardCharsets.UTF_8))) {
      Enumeration<JarEntry> es=jf.entries(); int count=0;
      while(es.hasMoreElements()) { JarEntry e=es.nextElement(); if(e.isDirectory()||!e.getName().endsWith(".class")||e.getName().equals("module-info.class")) continue;
        try(InputStream in=jf.getInputStream(e)) { ClassNode c=new ClassNode(); new ClassReader(in).accept(c,ClassReader.SKIP_DEBUG|ClassReader.SKIP_FRAMES);
          w.println("CLASS|"+c.name+"|"+(c.superName==null?"":c.superName));
          for(MethodNode m:c.methods) { w.println("METHOD|"+c.name+"|"+m.name+"|"+m.desc);
            int instructionIndex=0;
            for(AbstractInsnNode ins=m.instructions.getFirst();ins!=null;ins=ins.getNext()) if(ins.getOpcode()>=0) {
              w.println("INSN|"+c.name+"|"+m.name+"|"+m.desc+"|"+instructionIndex+"|"+opcodeName(ins.getOpcode()));
              instructionIndex++;
            }
            int seen=0; for(AbstractInsnNode n=m.instructions.getFirst();n!=null&&seen<80;n=n.getNext()) { String v=valueOf(n); boolean safeString=!(n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof String && (v.indexOf('|')>=0||v.indexOf('\n')>=0||v.indexOf('\r')>=0||v.indexOf('\t')>=0)); if(v!=null&&safeString) { w.println("CONST|"+c.name+"|"+m.name+"|"+m.desc+"|"+(n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof String?"string":n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Long?"long":n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Float?"float":n instanceof LdcInsnNode && ((LdcInsnNode)n).cst instanceof Double?"double":"int")+"|"+v); seen++; } }
          }
          if(++count>=2500) break;
        }
      }
    }
  }
  private static void patch(String input,String output,String plan) throws Exception {
    List<String> lines=Files.readAllLines(Paths.get(plan),StandardCharsets.UTF_8); Map<String,byte[]> replacements=new HashMap<>(); int total=0;
    Map<String,List<String[]>> byClass=new LinkedHashMap<>();
    for(String line:lines) { if(line.trim().isEmpty()||line.startsWith("#")) continue; String[] p=line.split("\\|",-1);
      if(p.length==8&&p[0].equals("PATCH")) byClass.computeIfAbsent(p[1],k->new ArrayList<>()).add(p);
      else if(p.length==7&&p[0].equals("OPCODE")) byClass.computeIfAbsent(p[1],k->new ArrayList<>()).add(p);
      else throw new IllegalArgumentException("Bad patch row: "+line);
    }
    if(byClass.isEmpty()) throw new IllegalArgumentException("Patch plan contains no PATCH or OPCODE rows");
    try(JarFile jf=new JarFile(input); JarOutputStream jos=new JarOutputStream(new BufferedOutputStream(Files.newOutputStream(Paths.get(output))))) {
      Enumeration<JarEntry> es=jf.entries(); Set<String> names=new HashSet<>();
      while(es.hasMoreElements()) { JarEntry e=es.nextElement(); String name=e.getName(); if(!names.add(name)) continue;
        String upper=name.toUpperCase(Locale.ROOT);
        if(upper.startsWith("META-INF/")&&(upper.endsWith(".SF")||upper.endsWith(".RSA")||upper.endsWith(".DSA")||upper.endsWith(".EC")||upper.startsWith("META-INF/SIG-"))) continue;
        byte[] data; try(InputStream in=jf.getInputStream(e)){ data=in.readAllBytes(); }
        if(name.endsWith(".class") && byClass.containsKey(name.substring(0,name.length()-6))) {
          ClassNode c=new ClassNode(); new ClassReader(data).accept(c,0);
          for(String[] p:byClass.get(c.name)) {
            String methodName=p[2], desc=p[3];
            MethodNode target=null; for(MethodNode m:c.methods) if(m.name.equals(methodName)&&m.desc.equals(desc)){ if(target!=null) throw new IllegalStateException("Ambiguous method "+c.name+"."+methodName+desc); target=m; }
            if(target==null) throw new IllegalStateException("Method not found: "+c.name+"."+methodName+desc);
            if(p[0].equals("OPCODE")) {
              int wanted=Integer.parseInt(p[4]); String oldOp=p[5], newOp=p[6];
              if(!safeOpcodePair(oldOp,newOp)) throw new IllegalArgumentException("Unsafe opcode pair: "+oldOp+" -> "+newOp);
              int index=0; boolean changed=false;
              for(AbstractInsnNode n=target.instructions.getFirst();n!=null;n=n.getNext()) if(n.getOpcode()>=0) {
                if(index++==wanted) {
                  if(!opcodeName(n.getOpcode()).equals(oldOp)) throw new IllegalStateException("Opcode mismatch at instruction "+wanted);
                  target.instructions.set(n, new JumpInsnNode(SAFE_OPCODES.get(newOp), ((JumpInsnNode)n).label)); changed=true; total++; break;
                }
              }
              if(!changed) throw new IllegalStateException("Instruction index not found: "+wanted);
            } else {
              String type=p[4], oldValue=p[5], newValue=p[6]; int occurrence=Integer.parseInt(p[7]);
              int index=0; boolean changed=false; for(AbstractInsnNode n=target.instructions.getFirst();n!=null;n=n.getNext()) {
                if(matchesType(n,type) && Objects.equals(valueOf(n),oldValue)) { if(index++==occurrence) { target.instructions.set(n,instruction(parse(type,newValue))); changed=true; total++; break; } }
              }
              if(!changed) throw new IllegalStateException("Constant not found: "+c.name+"."+methodName+desc+" old="+oldValue+" occurrence="+occurrence);
            }
          }
          ClassWriter cw=new ClassWriter(ClassWriter.COMPUTE_MAXS); c.accept(cw); data=cw.toByteArray(); replacements.put(name,data);
        }
        JarEntry outEntry=new JarEntry(name); outEntry.setTime(e.getTime()); jos.putNextEntry(outEntry); jos.write(data); jos.closeEntry();
      }
    }
    if(total==0) throw new IllegalStateException("No bytecode instructions were changed");
    System.out.println("PATCH_OK|"+total+"|"+replacements.size());
  }
  public static void main(String[] a) throws Exception {
    if(a.length==3&&a[0].equals("inspect")){inspect(a[1],a[2]);return;}
    if(a.length==4&&a[0].equals("patch")){patch(a[1],a[2],a[3]);return;}
    System.err.println("Usage: inspect <jar> <inventory.txt> | patch <in.jar> <out.jar> <plan.txt>"); System.exit(2);
  }
}
