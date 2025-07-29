# Tutorial: Executando os Notebooks de Teste de Hipóteses

## Introdução

Este repositório contém uma coleção de Jupyter Notebooks criados para fins educacionais, com foco em testes de hipóteses estatísticas.

## Pré-requisitos

Para executar os notebooks, você precisará ter o seguinte software instalado em seu sistema:

  * Python 3.8 ou superior
  * pip (gerenciador de pacotes do Python)
  * Jupyter Notebook ou JupyterLab

## Instalação

Estas instruções são para um sistema baseado em Debian, como o Ubuntu.

1.  **Instale o Python e o pip:**

    ```bash
    sudo apt update
    sudo apt install python3 python3-pip
    ```

2.  **Instale o Jupyter Notebook:**

    ```bash
    pip3 install jupyter
    ```

3.  **Instale as bibliotecas Python necessárias:**

    ```bash
    pip3 install numpy scipy statsmodels pandas matplotlib
    ```

## Executando os Notebooks

1.  Navegue até o diretório onde você salvou os arquivos `.ipynb` no seu terminal.

2.  Inicie o Jupyter Notebook com o seguinte comando:

    ```bash
    jupyter notebook
    ```

    Isso abrirá uma nova aba em seu navegador com a interface do Jupyter.

3.  Clique no nome do notebook que você deseja abrir e começar a explorar.

## Descrições dos Notebooks

1.  **1. Hypothesis intro.ipynb:** Apresenta os conceitos fundamentais do teste de hipóteses.
2.  **2. Hypothesis power function.ipynb:** Explora a função de poder de um teste de hipóteses.
3.  **3. Tests for One Sample.ipynb:** Demonstra testes de hipóteses para uma única amostra.
4.  **4. Tests for Two Samples.ipynb:** Cobre testes de hipóteses para comparar duas amostras.
5.  **5. Tests for Variance.ipynb:** Foca em testes para a variância de uma população.
6.  **6. Q-Q plots.ipynb:** Ensina a usar gráficos Q-Q para avaliar a normalidade dos dados.
7.  **7. Tests for paired data.ipynb:** Aborda testes para dados pareados.
8.  **8. Chi squared for categorical data.ipynb:** Explica o uso do teste qui-quadrado para dados categóricos.
9.  **9. Bootstrap.ipynb:** Introduz a técnica de bootstrap para inferência estatística.
10. **10. Multiarmed Bandits.ipynb:** Apresenta o problema dos "multi-armed bandits", uma aplicação de testes de hipóteses em aprendizado por reforço.


### Passo a Passo para Executar os Notebooks

1.  **Abra o Terminal**
    A primeira coisa a fazer é abrir o terminal do seu sistema. Você pode usar o atalho de teclado:
    `Ctrl + Alt + T`

2.  **Navegue até a Pasta do Projeto**
    Use o comando `cd` (change directory) para entrar na pasta onde você salvou os arquivos `.ipynb`. Por exemplo, se você salvou os notebooks em uma pasta chamada `testes-hipoteses` dentro dos seus Documentos, o comando seria:

    ```bash
    cd Documentos/testes-hipoteses/
    ```

    > **Dica**: Você pode usar o comando `ls` para listar os arquivos e diretórios e confirmar que você está na pasta correta. Você deverá ver a lista dos arquivos `.ipynb`.

3.  **Inicie o Jupyter Notebook**
    Uma vez dentro da pasta correta, execute o seguinte comando no terminal:

    ```bash
    jupyter notebook
    ```

    Ao executar este comando, você verá algumas mensagens de log no terminal, e uma nova aba deverá abrir automaticamente no seu navegador de internet padrão. Essa aba é a interface do Jupyter, mostrando todos os arquivos da pasta atual.

4.  **Abra um Notebook**
    Na interface do Jupyter no seu navegador, clique no nome de um dos arquivos `.ipynb` para abri-lo. Por exemplo, clique em `1. Hypothesis intro.ipynb`.

5.  **Execute as Células de Código**
    Um notebook é composto por **células** (cells). Existem células de texto (como esta que você está lendo) e células de código. Para "rodar" o notebook, você precisa executar as células de código.

      * Clique em uma célula de código para selecioná-la.
      * Para executar o código na célula selecionada, pressione **`Shift + Enter`**.

    Isso executará o código da célula atual e selecionará automaticamente a próxima.

    > **Importante**: Execute as células na ordem em que aparecem, de cima para baixo. Muitas vezes, uma célula de código depende de variáveis ou funções que foram definidas em uma célula anterior. Se você rodar fora de ordem, poderá encontrar erros.

Continue pressionando `Shift + Enter` para avançar pelo notebook, executando cada célula e observando os resultados que aparecem logo abaixo delas.
