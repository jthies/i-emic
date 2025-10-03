<ParameterList name="Trilinos THCM"><!--{-->
    
  <!-- Nonlinear solver parameters { -->
  <ParameterList name="NOX">

    <!-- "Line Search Based" or "Trust Region Based" (not tested) -->
    <Parameter name="Nonlinear Solver" type="string" value="Line Search Based"/>
    <!-- convergence tolerance for the Newton process -->
    <Parameter name="Convergence Tolerance" type="double" value="1.0e-6"/>

    <!-- Line Search parameters { -->
    <ParameterList name="Line Search">
      <!-- "Full Step" is standard Newton's, "Backtrack" is a line search, -->
      <!-- others exist as well -->
      <Parameter name="Method" type="string" value="Backtrack"/>
      
      <!-- maximum number of Newton iterations -->
      <Parameter name="Max Iters" type="int" value="10"/>
      
      <ParameterList name="Backtrack">
        <Parameter name="Default Step" type="double" value="1.0"/>
        <Parameter name="Max Iters" type="int" value="10"/>
        <Parameter name="Minimum Step" type="double" value="1e-6"/>
        <Parameter name="Recovery Step" type="double" value="1e-3"/>
      </ParameterList>
    </ParameterList><!-- } Line Search -->
    
    <!-- Parameters for the actual Newton solver { -->
    <ParameterList name="Direction">
      <Parameter name="Method" type="string" value="Newton"/>
      
      <!-- { -->
      <ParameterList name="Newton">
        
        <!-- Method to determine convergence tolerance for linear solver -->
        <!-- "Constant", "Type 1" (fails!) or "Type 2" -->
        <Parameter name="Forcing Term Method" type="string" value="Type 2"/>
        <Parameter name="Forcing Term Initial Tolerance" type="double" value="1.0e-2"/>
        <Parameter name="Forcing Term Maximum Tolerance" type="double" value="1.0e-2"/>
        <Parameter name="Forcing Term Minimum Tolerance" type="double" value="1.0e-5"/>
        
        <!-- Use Newton step even if linear solve failed (default 1)-->
        <!-- If you say "0" here the continuation run _stops_ when  -->
        <!-- the linear solver fails to achieve the requested tol   -->
        <Parameter name="Rescue Bad Newton Solve" type="bool" value="1"/>
        
        <!-- Note: the docu for this list is in class NOX::Epetra::LinearSystemAztecOO { -->
        <ParameterList name="Linear Solver">

          <!-- "GMRES", maybe others will be added        -->
          <Parameter name="Belos Solver" type="string" value="GMRES"/>

          <ParameterList name="Belos"><!--{-->
            <Parameter name="Flexible Gmres" type="bool" value="1"/>
            <Parameter name="Block Size" type="int" value="1"/>
            <Parameter name="Num Blocks" type="int" value="10"/>
            <Parameter name="Maximum Iterations" type="int" value="10"/>
            <Parameter name="Maximum Restarts" type="int" value="0"/>
            <Parameter name="Orthogonalization" type="string" value="DGKS"/>
            <Parameter name="Output Frequency" type="int" value="1"/>
          </ParameterList><!-- } Belos -->
          
          <!-- choose a preconditioner for the linear solves during Newton steps        -->
          <!-- "User Defined": Block GS/ILU or (see 'Block Preconditioner' sublist)     -->
          <!-- "New Ifpack": incomplete factorizations, parameters are set below        -->
          <!-- "ML": algebraic multigrid, parameters are set below                      -->
          <Parameter name="Preconditioner" type="string" value="New Ifpack"/>
          
          <!-- can be "Reuse", "Rebuild" or "Recompute"        -->
          
          <!-- "Rebuild" and "Recompute" are the same for most methods.           -->
          <!-- "Reuse" keeps the preconditioner for k linear system solves.       -->
          <!-- If you want to reuse the preconditioner at all, set this to "Reuse"-->
          <Parameter name="Preconditioner Reuse Policy" type="string" value="Rebuild"/>
          
          <!-- recompute preconditioner after K linear system solves. -->
          
          <!-- We have our own reuse-policy, controlled by the option -->
          <!-- with the same name in thcm_params.xml. If you want to  -->
          <!-- reuse the preconditioner, it's best to set this to -2  -->
          <!-- and use that option to control the maximum age.        -->
          <Parameter name="Max Age Of Prec" type="int" value="1"/>
          
          <!-- Ifpack Preconditioner for complete system: { --> 

          <!-- can be "ILU","ILUT","Amesos" and others    -->
          <!-- Only relevant if you chose "Ifpack" above. -->
          <Parameter name="Ifpack Preconditioner" type="string" 
                         value="Amesos stand-alone"/>

          <ParameterList name="Ifpack"><!--{-->

            <!-- if you chose Amesos above, select the direct solver here:            -->
            <!-- "KLU" is always available                                            -->
            <!-- "Superlu", "Mumps", "Mumps" depending on Trilinos installation -->
            <Parameter name="amesos: solver type" type="string" value="Mumps"/>
              
            <!-- general Amesos settings -->
            <Parameter name="OutputLevel" type="int" value="2"/>
            <Parameter name="PrintTiming" type="bool" value="1"/>
            <Parameter name="PrintStatus" type="bool" value="1"/>
            <Parameter name="NoDestroy" type="bool" value="0"/>
                
            <!-- this is how to set MUMPS parameters in Amesos. See the  -->
            <!-- MUMPS user's guide for details                          -->
            <!-- In general the default settings made by Amesos are ok   -->
            <ParameterList name="mumps">
              <!-- error stream: stdout=6, none=-1 -->
              <Parameter name="ICNTL(1)" type="int" value="6"/>
              <!-- info stream: stdout=6, none=-1 -->
              <Parameter name="ICNTL(2)" type="int" value="-1"/>
              <!-- global info stream: stdout=6, none=-1 -->
              <Parameter name="ICNTL(3)" type="int" value="6"/>
              <!-- verbosity (-1..3) -->
              <Parameter name="ICNTL(4)" type="int" value="1"/>
              <!-- collect statistics (for optimal performance set it to 0!) -->
              <Parameter name="ICNTL(11)" type="int" value="1"/>
               
              <!--Parameter name="CNTL(1)" type="double" value="?"/-->
            </ParameterList><!--}-->
          </ParameterList><!-- } global Ifpack -->
        </ParameterList><!-- } Linear Solver -->
      </ParameterList><!-- } Newton -->
    </ParameterList><!-- } direction -->    
  </ParameterList><!-- } NOX -->
</ParameterList> <!-- } -->
