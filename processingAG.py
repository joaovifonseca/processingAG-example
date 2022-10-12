"""
Implementação do Algoritmo Genético (AG) conforme teoria apresentada nas aulas 
Material de referência: IA_Algoritmos_Genéticos_2020-1-Alterado_04-05-2020.pptx
Autor: Sidnei Alves de Araújo 
Data: 08/05/2020
Disciplina: Inteligência Artificial
"""
"""
********************************** Importação de bibliotecas ****************************************
"""
import numpy as np
import timeit

"""
********************************** Lista de Variáveis Globais ***************************************
"""
"""
Modelagem do cromossomo 
"""
"""
Modelagem do cromossomo 
"""
#Colunas
#Preço, Volume, Agressor e Classificador
NUM_VAR = 4         #número de variáveis
L_INF = [5493, 5, 0, 0]           #limite(s) inferior(e) das variáveis
L_SUP = [5549.5, 4260, 1, 1]           #limite(s) inferior(e) das variáveis
TAM_CROMO = 40        #Número de bits do cromossomo
FENOTIPO = [10, 10, 10, 10]        #Número de bits de cada gene
"""
#Parametrização do AG
"""
TAM_POP = 300        #Tamanho da população - Número de indivíduos (soluções)
POS_PONTO_CORTE = 18  #Posição do ponto de corte (será usado no cruzamento)
PERC_MUT = 50        #Percentual de mutação (define o número de alelos mutados)
NUM_INDIV_SELEC = 30  #Número de indivíduos selecionados em cada geração
NUM_INDIV_ELITE = 1  #Número de indivíduos da elite (permancerão intactos na próxima geração)
TIPO_FO = 'Max'         #Tipo de função objetivo (FO): 'min' ou 'max'
NUM_GER = 100          #Número de gerações
NGSM = 10             #Número de gerações sem melhoria (para ser usado como critério de parada)
CRIT_PARADA=['ngsm', 10]   #Critério de parada do AG: ['fo',valor] ou ['num_ger',NUM_GER] ou ['ngsm',NGSM]





"""
#['fo', valor]        : encerra a evolução do AG com base no valor de FO que determina a solução ótima   
#['num_ger', NUM_GER] : encerra a evolução do AG com base no número de gerações
#['ngsm', NGSM]       : encerra a evolução do AG com base no número de gerações sem melhoria
"""   
fitness=[]

"""    
*****************************************************************************************************
Bloco do Algoritmo Genético (AG) - Classe ag
*****************************************************************************************************
"""
  
class ag:  
  """    
  ************************************* Funções Base do AG ******************************************
  """
  #Gera a população inicial de forma aleatória
  def gera_pop_incial(tam_pop, tam_crom):
      pop_ini=np.random.randint(0,2,[tam_pop,tam_crom])
      return pop_ini

  #Seleção dos melhores indivíduos
  def selecao(pop, num_indiv_selec, fitness, tipo_fo):
      # seleciona os [num_selec] melhores indivíduos da popução [pop] com base na aptidão [fitness].
      if tipo_fo[0:3].upper() != 'MAX' and tipo_fo[0:3].upper() != 'MIN' :
        print(tipo_fo, " não é um tipo de Função Objetivo conhecido! Use \'Min\' ou \'Max\'")
        return np.array([])
      if num_indiv_selec > TAM_POP:
        print("O número de indivíduos selecionados não pode ser maior que o tamanho da população!")
        return np.array([])
      #inicializa a matriz para receber os pais selecionados  
      pais_selec = np.zeros((num_indiv_selec, TAM_CROMO),'int')
      for ind_pai in range(num_indiv_selec):
          if tipo_fo[0:3].upper() == 'MAX':
              ind_fitness_max = np.where(fitness == np.max(fitness))
              fitness[ind_fitness_max] = -np.inf
          else:
              ind_fitness_max = np.where(fitness == np.min(fitness))
              fitness[ind_fitness_max] = +np.inf
          ind_fitness_max = ind_fitness_max[0][0]
          pais_selec[ind_pai, :] = pop[ind_fitness_max, :]
      return pais_selec

  #Cruzamento dos pais selecionados para recompor a população, como mostrado no slide 10 
  def cruzamento(pais_selec, pos_ponto_corte):
      #incializa matriz para receber os novos filhos gerados a partir do cruzamento dos pais selecionados
      pop_cruz = np.zeros((TAM_POP, TAM_CROMO),'int')
      if pos_ponto_corte > TAM_CROMO-1:
          print("Atenção! O ponto de corte é maior que o comprimento do cromossomo!")
          #se isso ocorrer, reposiciona o ponto de corte no meio do cromossomo 
          pos_ponto_corte = TAM_CROMO//2
      #inicia o "povoamento" da população (pop_cruz) a partir do cruzamento dos indivíduos selecionados  
      i=0
      #se quiser recompor apenas parte da população pelo cruzamento, trocar TAM_POP por PERC_CRUZ/100*TAM_POP   
      num_indiv_cruz = TAM_POP                 
      num_indiv_selec = NUM_INDIV_SELEC 
      for j in range(num_indiv_selec):
          for k in range(num_indiv_selec):
              if j != k:
                  #print(i,"0:",pos_ponto_corte,",",pos_ponto_corte,":", pais_selec.shape[1])
                  pop_cruz[i, 0:pos_ponto_corte] = pais_selec[j, 0:pos_ponto_corte]
                  pop_cruz[i, pos_ponto_corte:TAM_CROMO] = pais_selec[k, pos_ponto_corte:TAM_CROMO]
                  #print(pop_cruz[i,:])
                  i+=1
                  if i>=num_indiv_cruz:
                      break;
          if i>=num_indiv_cruz:
              break;
      #se após o cruzamento não foi possível gerar TAM_POP indivíduos, 
      #completa a população pop_cruz com indiv gerados aleatoriamente
      if i < TAM_POP-1:
          for l in range(i,TAM_POP):
              for c in range(TAM_CROMO):
                  pop_cruz[l][c]=np.random.randint(0,2)
      return pop_cruz

  #Mutação de alelos aleatórios dos indivíduos da população (ver slide 10)
  def mutacao(pop, perc_mut):
      pop_mut = pop
      tx_mut = perc_mut / 100
      if perc_mut > 100:
          print("Atenção! O percentual de mutação não pode ser maior que 100% e será ajustado para 50%")
          tx_mut = 0.5
      num_alelos_mut = tx_mut * TAM_POP * TAM_CROMO
      #print("num_alelos_mut = ",num_alelos_mut)
      for i in range(int(num_alelos_mut)):
          #print(ini, int(pop.shape[0]))
          l = np.random.randint(0,TAM_POP)
          c = np.random.randint(0,TAM_CROMO)
          if pop_mut[l][c] == 0:
              pop_mut[l][c] = 1
          else:
              pop_mut[l][c] = 0
      return pop_mut

  #Insere na proxima geração parte dos indivíduos selecionados na geração corrente
  #Essa operação é opcional no AG, sendo controlada pela variável NUM_INDIV_ELITE
  def elitismo(pais_selc, pop, num_indiv_elite):
      #verifica se o número de indivíduos da elite for maior que o número de indiv selecionados  
      num_indiv_elite = NUM_INDIV_ELITE
      if(num_indiv_elite > NUM_INDIV_SELEC):
          num_indiv_elite = NUM_INDIV_SELEC // 2
      num_indiv_pop = TAM_POP - num_indiv_elite
      #agrupa parte da dos indivídos selecionados com os n primeiros indivídos da população corrente
      #num_indiv_elite + num_indiv_pop = TAM_POP
      pop_com_elite = np.append(pais_selc[0:num_indiv_elite, :] , pop[0:num_indiv_pop, :], axis=0)
      #retorna a população com os indivíduos da elite inseridos no início
      return pop_com_elite
      
  """
  ***************************************************************************************************
  Funções diversas
  ***************************************************************************************************
  """
  # esquema de decodificação mostrado no slide 15 - conversão de base 2 para base 10 
  def conv_bin2dec(b):
      dec=0
      pot=0
      for i in range(len(b)-1,-1,-1):
          dec+=b[i] * 2**pot
          pot=pot+1
      return dec

  # esquema de decodificação mostrado no slide 16 - binário para Número Inteiro (Z) no intervalo [inf,sup]
  def conv_bin2int(b, l_inf, l_sup):
      v=ag.conv_bin2dec(b)
      k = b.shape[0]
      v_int = l_inf + ( (l_sup - l_inf) / (2**k -1) ) * v  
      return int(v_int)

  # esquema de decodificação mostrado no slide 16 - binário para Número Real (R) no intervalo [inf,sup]
  def conv_bin2real(b, l_inf, l_sup):
      v=ag.conv_bin2dec(b)
      k = b.shape[0]
      v_real = l_inf + ( (l_sup - l_inf) / (2**k -1) ) * float(v)  
      return v_real

  # Calcula o tamanho do cromossomo (k) conforme mostrado nos slides 17 e 18 
  def calc_tam_gene(l_inf, l_sup, p):
      calc = (l_sup - l_inf) * 10**p
      k = np.log2(calc) 
      k = np.math.floor(k)+1
      return k
  """
  ***************************************************************************************************
  Ciclo de evolução do AG, conforme fluxograma do slide 9
  ***************************************************************************************************
  """
      
  def evolucao(func_aptidao):
      tp_ini = timeit.default_timer()
      cont_ger_sem_melhoria = 0
      crit_parada = CRIT_PARADA
      crit_parada_atingido = False            
      tipo_fo=TIPO_FO.upper()
      print("Resolvendo um problema de otimzação do tipo: ", tipo_fo)
      ind_melhor_fitness=0
      if tipo_fo == 'MIN':
          melhor_fitness = np.inf
      else:
          melhor_fitness = -np.inf
          
      #Gera a população inicial (de forma aleatória)
      pop = ag.gera_pop_incial(TAM_POP, TAM_CROMO)

      cont_ger=0
      print("Entrando no ciclo de evolução do AG")
      #enquando o critério de parada não é atingido 
      print("Geração: ", end="")
     
      while(crit_parada_atingido == False ):    
          #incrementa 1 no contador de gerações
          cont_ger+=1
          print(cont_ger,"\b, ", end="")
          print("\n")
          
          #cria um vetor para armazenar o fitness de cada indivíduo 
          fitness = np.array(np.zeros(TAM_POP))
          
          
          
          #Calcula a aptidão de cada indivíduo com base na função de fitness
          for i in range(TAM_POP):
              fitness[i] = func_aptidao(pop[i])
              #print("individuo: ", pop[i], ". aptidão: ", fitness[i])
              
          
          #verifica o tipo de problema para avaliar o fitness (aptidão)
          #se é um problema de minimização
          if tipo_fo == 'MIN':
              #se houve melhoria, então o contador de gerações sem melhoria deve ser zerado
              if np.min(fitness) < melhor_fitness:
                  ind_melhor_fitness = np.where(fitness == np.min(fitness))[0][0]
                  melhor_fitness = fitness[ind_melhor_fitness]
                  melhor_indiv = pop[ind_melhor_fitness,:]
                  cont_ger_sem_melhoria = 0
              else:
                  cont_ger_sem_melhoria += 1
          else: #se não é problema de maximização
              #se houve melhoria, então o contador de gerações sem melhoria deve ser zerado
              if np.max(fitness) > melhor_fitness:
                  ind_melhor_fitness = np.where(fitness == np.max(fitness))[0][0]
                  melhor_fitness = fitness[ind_melhor_fitness]
                  melhor_indiv = pop[ind_melhor_fitness,:]
                  cont_ger_sem_melhoria = 0
              else:
                  cont_ger_sem_melhoria += 1
          
          #Verifica critério de parada
          if crit_parada[0].upper() == 'FO':
              for ind in range(len(fitness)):
                  if fitness[ind] == crit_parada[1] or cont_ger_sem_melhoria > NGSM: 
                    crit_parada_atingido = True 
                    break
          elif crit_parada[0].upper() == 'NGSM':
              if crit_parada[1] == cont_ger_sem_melhoria: 
                  crit_parada_atingido = True 
          else:
              if crit_parada[1] == cont_ger:
                  crit_parada_atingido = True 
                  
          #se atingiu o critério de parada encerra o AG e retorna a melhor solução encontrada        
          if crit_parada_atingido == True: 
              print("\b\b.\n")
              tp_fim = timeit.default_timer()
              print("Tempo de processamento = %.3f segundos\n" % float(tp_fim-tp_ini))
              return [cont_ger, melhor_indiv, melhor_fitness]
          
          #Seleção dos melhores indivíduos da população pop
          pais_selec = ag.selecao(pop, NUM_INDIV_SELEC, fitness, TIPO_FO)

          #Aplica o opreador de cruzamento para recompor a população
          pop_cruz = ag.cruzamento(pais_selec, POS_PONTO_CORTE)

          #Aplica o operador de mutação para gerar diversidade populacional   
          pop_mut = ag.mutacao(pop_cruz, PERC_MUT)

          #aplica elitismo (indivíduos a serem evados intactos para a proxima geração)
          if NUM_INDIV_ELITE > 0:
              pop_eli = ag.elitismo(pais_selec, pop_mut, NUM_INDIV_ELITE)   
              pop = pop_eli
          else:
              pop = pop_mut            
          
      #Encerrado o ciclo de evolução do AG, apresenta-se o resultado
      return [cont_ger, melhor_indiv, melhor_fitness]

"""    
*****************************************************************************************************
Fim da Classe ag - Daqui para baixo vem a modelagem do problema a ser resolvido: modelagem do cromos-
somo, parametrização do AG e definição da FO que será usada pelo AG para avaliação das soluções  
*****************************************************************************************************
"""


# calcular a função aptidão

def calc_aptidao(cromossomo):
    # separa os genes conforme a informação do fenótipo
    gene1=cromossomo[0:5]
    gene2=cromossomo[5:10]
    #faz a decodificação
    x1 = ag.conv_bin2dec(gene1)
    x2 = ag.conv_bin2dec(gene2)
    #calcula o valor da função objetivo
    fo = 30 * x1 + 40 * x2
    #se qualquer uma das três restrições for violada, troca o sinal da fo
    if x1 > 24 or x2 > 16 or x1 + 2*x2 > 40:
        fo = fo * -1
    return fo

#chama o AG com base na parametrização
res = ag.evolucao(calc_aptidao)

#exibe a solução encontrada
print('Após', res[0],'gerações, o AG encontrou a seguinte solução:')
print('Cromossomo: ', res[1])
print('Fitness: ', res[2])
print('Radios Standard..(x1) = ', ag.conv_bin2dec(res[1][0:5]))
print('Radios Luxo......(x2) = ', ag.conv_bin2dec(res[1][5:10]))
