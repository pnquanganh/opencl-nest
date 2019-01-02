//------------------------------------------------------------------------------
// Tooling sample. Demonstrates:
//
// * How to write a simple source tool using libTooling.
// * How to use RecursiveASTVisitor to find interesting AST nodes.
// * How to use the Rewriter API to rewrite the source code.
//
// Eli Bendersky (eliben@gmail.com)
// This code is in the public domain
//------------------------------------------------------------------------------
#include <sstream>
#include <string>
#include <iostream>
#include <inttypes.h>
#include <map>
#include <set>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;
using namespace std;

static llvm::cl::OptionCategory ToolingSampleCategory("Tooling Sample");

class DataInfo {
public:
  string type;
  string host_id;
  string gpu_id;
  bool isScalar;
  uint64_t array_size;
  bool isOutput;

  DataInfo(): type(""), host_id(""), gpu_id(""), isScalar(true), array_size(0), isOutput(false)
  {};

  DataInfo(string &type_, string &host_id_, string &gpu_id_, bool isScalar_, uint64_t array_size_, bool isOutput_ = false) : type(type_), host_id(host_id_), gpu_id(gpu_id_), isScalar(isScalar_), array_size(array_size_), isOutput(isOutput_)
  {};

  string toString() {
    string res = "";

    res += type + "\n";
    res += host_id + "\n";
    res += gpu_id + "\n";
    res += string(isScalar ? "True" : "False") + "\n";
    // char *tmp[10];
    // sprintf(tmp, "%" PRId64 "\n", array_size);
    res += to_string(array_size) + "\n";
    res += string(isOutput ? "True" : "False") + "\n";
    return res;
  }
};

typedef map<string, DataInfo> data_map;

// By implementing RecursiveASTVisitor, we can specify which AST nodes
// we're interested in by overriding relevant methods.
class MyASTVisitor : public RecursiveASTVisitor<MyASTVisitor> {
public:
  MyASTVisitor(SourceManager &sm_, CompilerInstance &ci_, Rewriter &R, data_map &D, set<string> &RB) : TheRewriter(R), sm(sm_), ci(ci_), visitingUpdate(false), visitingDynamics(false), visitingForStmt(false), visitingHandle(false), VarData(D), RingBuffers(RB), eventId("") {}

  void processStmt(Stmt *currStmt) {

    if (!currStmt)
      return;
    

    // else if (isa<CompoundStmt>(currStmt))
    //   {
    // 	//printStmt(currStmt);
	  
    // 	CompoundStmt *cs = cast<CompoundStmt>(currStmt);
    // 	for (CompoundStmt::body_iterator iCs = cs->body_begin(),
    // 		 eCs = cs->body_end();
    // 	     iCs != eCs; ++iCs)
    // 	  {
    // 	    Stmt *currCompoundStmt = *iCs;
	    
    // 	    processStmt(currCompoundStmt);
    // 	  }
    //   }
    // else if (isa<BinaryOperator>(currStmt))
    //   {
	
    // 	BinaryOperator *bo = cast<BinaryOperator>(currStmt);
    // 	Stmt *lhs = cast<Stmt>(bo->getLHS());
    // 	Stmt *rhs = cast<Stmt>(bo->getRHS());
    // 	processStmt(lhs);
    // 	StringRef s = BinaryOperator::getOpcodeStr(bo->getOpcode());
    // 	printf(" %s ", s.str().c_str());
	    
    // 	processStmt(rhs);
    // 	if (!insideCond)
    // 	  printf(";\n");
    //   }
    // else if (isa<MemberExpr>(currStmt))
    //   {
    // 	printStmt(currStmt);
    //   }
    

  }

  void insertTextBefore(FunctionDecl *s, const char *t)
  {
    stringstream SSBefore;
    SSBefore << t;
    SourceLocation ST = s->getQualifierLoc().getEndLoc().getLocWithOffset(0);
    TheRewriter.InsertText(ST, SSBefore.str(), true, true);

  }
  
  void insertTextBefore(Stmt *s, const char *t)
  {
    stringstream SSBefore;
    SSBefore << t;
    SourceLocation ST = s->getLocStart().getLocWithOffset(0);
    TheRewriter.InsertText(ST, SSBefore.str(), true, true);
  }

  void insertTextAfter(Stmt *s, const char *t)
  {
    stringstream SSAfter;
    SSAfter << t;
    SourceLocation ST = s->getLocEnd().getLocWithOffset(1);
    TheRewriter.InsertText(ST, SSAfter.str(), true, true);
  }
  
  void processWhileStmt(Stmt *s) {
    
    WhileStmt *While = cast<WhileStmt>(s);

    insertTextBefore(While, "/*GSL CALL*/\n/*");

    Stmt *WhileBody = While->getBody();

    insertTextAfter(WhileBody, "*/\n");
  }
  
  void processForStmt(Stmt *s) {

    ForStmt *f = cast<ForStmt>(s);

    Stmt *ForBody = f->getBody();

    insertTextBefore(ForBody, "/*GPU START*/\n");
    insertTextAfter(ForBody, "/*GPU END*/\n");
    
  }
  void printStmt(Stmt *s) {
    bool invalid;
    
    CharSourceRange conditionRange =
      CharSourceRange::getTokenRange(s->getLocStart(), s->getLocEnd());
    StringRef str =
      Lexer::getSourceText(conditionRange, sm, ci.getLangOpts(), &invalid);
    printf("%s", str.str().c_str());
  }


  void processArraySubcript(Stmt *currStmt)
  {
    ArraySubscriptExpr *arr = cast<ArraySubscriptExpr>(currStmt);
    Expr *rhs = arr->getRHS();
    if (isa<DeclRefExpr>(rhs))
      {
	DeclRefExpr *dre = cast<DeclRefExpr>(rhs);
	string name = dre->getNameInfo().getAsString();
	string s = name + "*num_nodes + tid";
	//printf("dre: %s\n", name.c_str());
	TheRewriter.ReplaceText(dre->getSourceRange(), StringRef(s));
      }
  }

  void processMemberExpr(MemberExpr *me, bool is_gsl_arg)
  {
    string s = "";
    string host_id = "";
    string gpu_id = "";

    MemberExpr *expr = me;
    while (true)
      {
	string name = expr->getMemberNameInfo().getAsString();

	if (s != "")
	  {
	    s = "_" + s;
	    host_id = "." + host_id;
	  }
	s = name + s; //str.str();
	host_id = name + host_id;
		
	Expr *base = expr->getBase();
	if (not isa<MemberExpr>(base))
	  break;
	expr = cast<MemberExpr>(base);
      }

    gpu_id = s;
    string s_type = ""; //me->getMemberDecl()->getType().getAsString();
    uint64_t array_size = 0;

    const Type *t = me->getMemberDecl()->getType().getTypePtr();
	    
    bool check_type = false;
    bool is_scalar = true;
    if (isa<BuiltinType>(t))
      {
	s = s + "[tid]";
	check_type = true;
	s_type = me->getMemberDecl()->getType().getAsString();
	if (s_type == "_Bool")
	  s_type = "bool";
      }
    else if (isa<ConstantArrayType>(t))
      {
	is_scalar = false;
	const ConstantArrayType *cat = cast<ConstantArrayType>(t);
	const Type *b_type = cat->getElementType().getTypePtr();
	array_size = cat->getSize().getZExtValue();

	if (isa<BuiltinType>(b_type))
	  {
	    check_type = true;
	    s_type = cat->getElementType().getAsString();
	    if (s_type == "_Bool")
	      s_type = "bool";
	  }
      }

    if (check_type)
      {
	TheRewriter.ReplaceText(me->getSourceRange(), StringRef(s));
	data_map::iterator it_map = VarData.find(host_id);
	
	if (it_map == VarData.end())
	  {
	    DataInfo curr_var(s_type, host_id, gpu_id, is_scalar, array_size, is_gsl_arg);
	    VarData.insert(pair<string, DataInfo>(host_id, curr_var));
	  }
	else if (is_gsl_arg)
	  {
	    it_map->second.isOutput = true;
	  }
      }
  }
  
  void processUpdateStmt(Stmt *currStmt)
  {
    if (isa<ForStmt>(currStmt))
      {
	processForStmt(currStmt);
      }
    // else if (isa<WhileStmt>(currStmt))
    //   {
    // 	processWhileStmt(currStmt);
    //   }
    else if (isa<CXXMemberCallExpr>(currStmt))
      {
	CXXMemberCallExpr *caller = cast<CXXMemberCallExpr>(currStmt);
	Expr *callee = caller->getCallee();
	if (isa<MemberExpr>(callee))
	  {
	    MemberExpr *callee_expr = cast<MemberExpr>(callee);
	      
	    string name = callee_expr->getMemberNameInfo().getAsString();
	    if (name.find("set_spiketime") != string::npos)
	      {
		insertTextBefore(currStmt, "/*SPIKE SEND*/\n/*");
	      }
	    else if (name.find("send") != string::npos)
	      {
		Expr *base = callee_expr->getBase();
		if (base && isa<MemberExpr>(base))
		  {
		    MemberExpr *base_memb = cast<MemberExpr>(base);
		    name = base_memb->getMemberNameInfo().getAsString();
		    if (name.find("event_delivery_manager") != string::npos)
		      {
			insertTextAfter(currStmt, "*/\n");
		      }
		  }
	      }
	    else if (name.find("get_value") != string::npos)
	      {
		Expr *base = callee_expr->getBase();
		
		if (base && isa<MemberExpr>(base))
		  {
		    MemberExpr *base_memb = cast<MemberExpr>(base);
		    string rb_name = base_memb->getMemberNameInfo().getAsString();
		    string s_type = base_memb->getMemberDecl()->getType().getAsString();
		    if (s_type.find("nest::RingBuffer") != string::npos)
		      {
			string rb_stmt =
			  "ring_buffer_get_value(" + rb_name + ", ring_buffer_size, num_nodes, tid, lag, time_index)";
			TheRewriter.ReplaceText(currStmt->getSourceRange(), StringRef(rb_stmt));
			RingBuffers.insert(rb_name);
		      }
		  }
		
	      }
	    else if (name.find("record_data") != string::npos)
	      {
		insertTextBefore(currStmt, "/*LOG DATA*/\n//");
	      }
	  }
      }
    else if (isa<ArraySubscriptExpr>(currStmt))
      {
	processArraySubcript(currStmt);
      }
    else if (isa<MemberExpr>(currStmt))
      {
	MemberExpr *me = cast<MemberExpr>(currStmt);
	processMemberExpr(me, false);
      }
    else if (isa<CallExpr>(currStmt))
      {
	CallExpr *gsl_call = cast<CallExpr>(currStmt);
	Expr *callee = gsl_call->getCallee();
	if (isa<ImplicitCastExpr>(callee))
	  {
	    ImplicitCastExpr* ice_callee = cast<ImplicitCastExpr>(callee);
	    Expr *callee_sub_expr = ice_callee->getSubExpr();
	    if (isa<DeclRefExpr>(callee_sub_expr))
	      {
		DeclRefExpr *dre = cast<DeclRefExpr>(callee_sub_expr);
		string name = dre->getNameInfo().getAsString();
		//printf("name: %s\n", name.c_str());
		if (name == "gsl_odeiv_evolve_apply")
		  {

		    insertTextBefore(currStmt, "/*GSL CALL*/\n/*");
		    insertTextAfter(currStmt, "*/\n");

		    int num_args = gsl_call->getNumArgs();
		    Expr *arg = gsl_call->getArgs()[num_args - 1];
	    
		    if (isa<ImplicitCastExpr>(arg))
		      {
			ImplicitCastExpr* ice = cast<ImplicitCastExpr>(arg);
			Expr *sub_expr = ice->getSubExpr();
			if (isa<MemberExpr>(sub_expr))
			  {
			    MemberExpr *me = cast<MemberExpr>(sub_expr);
			    processMemberExpr(me, true);
			  }
		      }
		  }
	      }
	  }
	    
	//printf("num_args: %d\n", num_args);
      }
    else if (isa<CXXThrowExpr>(currStmt))
      {
	TheRewriter.ReplaceText(currStmt->getSourceRange(), StringRef("return;"));
      }
  }

  void processDynamicsStmt(Stmt *currStmt)
  {
    if (isa<DeclStmt>(currStmt))
      {
	DeclStmt *decl_stmt = cast<DeclStmt>(currStmt);
	Decl *decl = decl_stmt->getSingleDecl();

	if (isa<TypedefDecl>(decl))
	  {
	    insertTextBefore(currStmt, "/*");
	  }
	else if (isa<VarDecl>(decl))
	  {
	    VarDecl *var_decl = cast<VarDecl>(decl);
	    string name = var_decl->getNameAsString();
	    
	    // const Type* type = var_decl->getTypeSourceInfo()->getTypeLoc().getTypePtr();
	    if (name == "node")
	      {
	    	insertTextAfter(currStmt, "*/\n");
	      }
	  }
      }
    else if (isa<MemberExpr>(currStmt))
      {
	MemberExpr *me = cast<MemberExpr>(currStmt);

	string s = "";
	string host_id = "";
	string gpu_id = "";

	MemberExpr *expr = me;
	bool check_node = false;
	while (true)
	  {
	    string name = expr->getMemberNameInfo().getAsString();

	    if (name != "node")
	      {
		if (s != "")
		  {
		    s = "_" + s;
		    host_id = "." + host_id;
		  }
		s = name + s; //str.str();
		host_id = name + host_id;
	      }
	    
	    Expr *base = expr->getBase();
	    if (not isa<MemberExpr>(base))
	      {
		if (isa<DeclRefExpr>(base))
		  {
		    DeclRefExpr *dre = cast<DeclRefExpr>(base);
		    string name = dre->getNameInfo().getAsString();
		    if (name == "node")
		      check_node = true;
		  }
		break;
	      }
	    expr = cast<MemberExpr>(base);
	  }

	if (not check_node)
	  return;

	// printf("%s\n", s.c_str());

	//ValueDecl *Decl = me->getMemberDecl();

	gpu_id = s;
	string s_type = ""; //me->getMemberDecl()->getType().getAsString();
	uint64_t array_size = 0;

	// s += " ";
	// s += s_type;
	// s += "\n";

	// printf("%s\n", s.c_str());
	const Type *t = me->getMemberDecl()->getType().getTypePtr();

	bool check_type = false;
	bool is_scalar = true;
	if (isa<BuiltinType>(t))
	  {
	    s = s + "[tid]";
	    check_type = true;
	    s_type = me->getMemberDecl()->getType().getAsString();
	    if (s_type == "_Bool")
	      s_type = "bool";
	  }
	else if (isa<ConstantArrayType>(t))
	  {
	    is_scalar = false;
	    const ConstantArrayType *cat = cast<ConstantArrayType>(t);
	    const Type *b_type = cat->getElementType().getTypePtr();
	    array_size = cat->getSize().getZExtValue();
	    
	    //b_type->dump();
	    if (isa<BuiltinType>(b_type))
	      {
		check_type = true;
		s_type = cat->getElementType().getAsString();
		if (s_type == "_Bool")
		  s_type = "bool";
	      }
	  }

	if (check_type)
	  {
	    //printf("%s\n", s.c_str());
	    TheRewriter.ReplaceText(me->getSourceRange(), StringRef(s));
	    if (VarData.find(host_id) == VarData.end())
	      {
		//printf("Dynamics: %s\n", host_id.c_str());
		DataInfo curr_var(s_type, host_id, gpu_id, is_scalar, array_size);
		VarData.insert(pair<string, DataInfo>(host_id, curr_var));
	      }
	  }

	// ASTContext& ctx = Decl->getASTContext();
	// SourceManager& mgr = ctx.getSourceManager();
	// SourceRange range = SourceRange(Decl->getSourceRange().getBegin(), Decl->getBody()->getSourceRange().getBegin());
	// StringRef sr = clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(range), mgr, ctx.getLangOpts());

	// printf("%s\n", sr.str().c_str());
	      
      }
    else if (isa<ArraySubscriptExpr>(currStmt))
      {
	processArraySubcript(currStmt);
      }
    else if (isa<DeclRefExpr>(currStmt))
      {
	DeclRefExpr *dre = cast<DeclRefExpr>(currStmt);
	string name = dre->getNameInfo().getAsString();
	
	const Type *type = dre->getType().getTypePtr();
	if (isa<FunctionType>(type))
	  {
	    //printf("name %s\n", name.c_str());
	    string qname = "";
	    if (dre->hasQualifier())
	      {
		IdentifierInfo *info = dre->getQualifier()->getAsIdentifier();
		if (info)
		  {
		    qname = info->getName().str();
		    //if (dre->getQualifier()->getAsIdentifier()->isStr(StringRef("std")))
		  }
		else
		  {
		    NamespaceDecl *Namespace = dre->getQualifier()->getAsNamespace();
		    if (Namespace)
		      {
			 qname = Namespace->getNameAsString();
		      }
		    else
		      {
			NamespaceAliasDecl *Alias = dre->getQualifier()->getAsNamespaceAlias();
			if (Alias)
			  {
			    qname = Namespace->getNameAsString();
			  }
		      }
		  }
	      }
	    if (qname == "std")
	      {
		
		//printf("qname %s\n", qname.c_str());
		TheRewriter.ReplaceText(currStmt->getSourceRange(), StringRef(name));
	      }

	  }
      }

  }

  void processHandleStmt(Stmt *currStmt)
  {
    if (isa<CXXMemberCallExpr>(currStmt))
      {
	CXXMemberCallExpr *caller = cast<CXXMemberCallExpr>(currStmt);
	Expr *callee = caller->getCallee();
	if (isa<MemberExpr>(callee))
	  {
	    MemberExpr *callee_expr = cast<MemberExpr>(callee);
	      
	    string name = callee_expr->getMemberNameInfo().getAsString();
	    if (name.find("add_value") != string::npos)
	      {
		Expr *base = callee_expr->getBase();
		if (base && isa<MemberExpr>(base))
		  {
		    MemberExpr *base_memb = cast<MemberExpr>(base);
		    string rb_name = base_memb->getMemberNameInfo().getAsString();
		    string rb_stmt = "ring_buffer_add_value(h_" + rb_name
		      + ", h_" + rb_name + "_mark, " + eventId + ".get_sender_gid() - 1";
		    for (unsigned int i = 0; i < caller->getNumArgs(); i++)
		      {
			Expr *e = caller->getArg(i);
			//SourceManager& mgr = ctx.getSourceManager();
			//SourceRange range = SourceRange(f->getSourceRange().getBegin(), f->getBody()->getSourceRange().getBegin());
			StringRef s = clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(e->getSourceRange()), sm, ci.getLangOpts());
			rb_stmt += ", " + s.str();
		      }
		    rb_stmt += ")";
		    TheRewriter.ReplaceText(currStmt->getSourceRange(), StringRef(rb_stmt));
		  }

	      }
	  }
      }
  }
  
  bool VisitStmt(Stmt *currStmt) {

    if (visitingUpdate && visitingForStmt)
      processUpdateStmt(currStmt);
    else if (visitingDynamics)
      processDynamicsStmt(currStmt);
    else if (visitingHandle)
      processHandleStmt(currStmt);
    return true;
  }

  bool dataTraverseStmtPre (Stmt *s) {

    if (visitingUpdate)
      {
	if (isa<ForStmt>(s))
	  {
	    visitingForStmt = true;
	  }
      }

    return true;
  }

  bool dataTraverseStmtPost (Stmt *s) {

    if (visitingUpdate)
      {
	if (visitingForStmt && isa<ForStmt>(s))
	  visitingForStmt = false;
      }
    return true;
  }

  bool VisitFunctionDecl(FunctionDecl *f) {
    // Only function definitions (with bodies), not declarations.
    if (f->hasBody()) {
      // Stmt *FuncBody = f->getBody();

      // Type name as string
      // QualType QT = f->getReturnType();
      // string TypeStr = QT.getAsString();

      // Function name
      //DeclarationName DeclName = f->getNameInfo().getName();
      
      string FuncName = f->getNameInfo().getAsString(); //DeclName.getAsString();
      
      // ASTContext& ctx = f->getASTContext();
      // SourceManager& mgr = ctx.getSourceManager();
      // SourceRange range = SourceRange(f->getSourceRange().getBegin(), f->getBody()->getSourceRange().getBegin());
      // StringRef s = clang::Lexer::getSourceText(clang::CharSourceRange::getTokenRange(range), mgr, ctx.getLangOpts());

      if (FuncName == "update")//(s.find("nest::") != string::npos && s.find("::update") != string::npos)
	{
	  // Stmt *FuncBody = f->getBody();
	  // // //Stmt *s = FuncBody;

	  // for (Stmt::child_iterator i = FuncBody->child_begin(), e = FuncBody->child_end(); i != e; ++i) {
	  //   Stmt *currStmt = *i;

	  //   if (isa<ForStmt>(currStmt))
	  //     {
	  // 	processForStmt(currStmt);
	  // 	break;
	  //     }

	  visitingUpdate = true;
	  TraverseStmt(f->getBody());
	  //TraverseDecl(f);
	  visitingUpdate = false;
	  //while (s != nullptr)
	    //printStmt(currStmt);

	  
	  //f->dump();

	  // insideUpdate = true;
	  // TraverseStmt(f->getBody()); //, &Queue);
	  // insideUpdate = false;

	  // printf("%s\n", s.str().c_str());
	  // //printf("%s %s\n", TypeStr.c_str(), FuncName.c_str());
	  // printf("------\n");
	  
	  //return true;
	}
      else if (FuncName.find("dynamics") != string::npos)
	{
	  visitingDynamics = true;
	  Stmt *dynamics_body = f->getBody();
	  insertTextBefore(dynamics_body, "/*DYNAMICS FUNCTION START*/");
	  insertTextAfter(dynamics_body, "/*DYNAMICS FUNCTION END*/");
	  TraverseStmt(dynamics_body);
	  visitingDynamics = false;
	}
      else if (FuncName == "handle") //(s.find("handle") != string::npos)
      	{
	  if (f->getNumParams() == 1)
	    {
	      ParmVarDecl *decl = cast<ParmVarDecl>(*(f->param_begin()));
	      eventId = decl->getNameAsString();
	      string s_type = decl->getType().getAsString();
	      if (s_type.find("Event") != string::npos)
		{
		  visitingHandle = true;
		  Stmt *handle_body = f->getBody();
		  insertTextBefore(f, "/*HANDLE START*/");
		  insertTextAfter(handle_body, "/*HANDLE END*/");
		  TraverseStmt(handle_body);
		  visitingHandle = false;
		}
	    }
      	}

      // // Add comment before
      // stringstream SSBefore;
      // SSBefore << "// Begin function " << FuncName << " returning " << TypeStr
      //          << "\n";
      // SourceLocation ST = f->getSourceRange().getBegin();
      // TheRewriter.InsertText(ST, SSBefore.str(), true, true);

      // // And after
      // stringstream SSAfter;
      // SSAfter << "\n// End function " << FuncName;
      // ST = FuncBody->getLocEnd().getLocWithOffset(1);
      // TheRewriter.InsertText(ST, SSAfter.str(), true, true);
    }

    return true;
  }

private:
  Rewriter &TheRewriter;
  SourceManager &sm; // Initialize me!
  CompilerInstance &ci;  // Initialize me!
  bool visitingUpdate;
  bool visitingDynamics;
  bool visitingForStmt;
  bool visitingHandle;
  data_map &VarData;
  set<string> &RingBuffers;
  string eventId;
};

// Implementation of the ASTConsumer interface for reading an AST produced
// by the Clang parser.
class MyASTConsumer : public ASTConsumer {
public:
  MyASTConsumer(SourceManager &sm_, CompilerInstance &ci_, Rewriter &R, data_map &D, set<string> &RB) : Visitor(sm_, ci_,R,D, RB) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      // Traverse the declaration using our AST visitor.
      Visitor.TraverseDecl(*b);
      //(*b)->dump();
    }
    return true;
  }

private:
  MyASTVisitor Visitor;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {

    printf("%lu\n", VarData.size());
    for (data_map::iterator it = VarData.begin(); it != VarData.end(); it++)
      {
	DataInfo &info = it->second;
	string s = info.toString();
	printf("%s", s.c_str());
	//printf("----\n");
      }

    printf("%lu\n", RingBuffers.size());
    for (set<string>::iterator it = RingBuffers.begin(); it != RingBuffers.end(); it++)
      {
	printf("%s\n", it->c_str());
      }

    fflush(stdout);
       
    SourceManager &SM = TheRewriter.getSourceMgr();
    // llvm::errs() << "** EndSourceFileAction for: "
    //              << SM.getFileEntryForID(SM.getMainFileID())->getName() << "\n";

    // Now emit the rewritten buffer.
    TheRewriter.getEditBuffer(SM.getMainFileID()).write(llvm::outs());

  }

  unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    //llvm::errs() << "** Creating AST consumer for: " << file << "\n";
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(CI.getSourceManager(), CI, TheRewriter, VarData, RingBuffers);
  }

private:
  Rewriter TheRewriter;
  data_map VarData;
  set<string> RingBuffers;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, ToolingSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  // ClangTool::run accepts a FrontendActionFactory, which is then used to
  // create new objects implementing the FrontendAction interface. Here we use
  // the helper newFrontendActionFactory to create a default factory that will
  // return a new MyFrontendAction object every time.
  // To further customize this, we could create our own factory class.
  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
