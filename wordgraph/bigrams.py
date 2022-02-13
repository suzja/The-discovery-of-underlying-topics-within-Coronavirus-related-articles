import sys
import pprint
import pandas as pd
from nltk.util import ngrams
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import re
import nltk
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
import re
import networkx as nx
import plotly.graph_objects as go
#Set up a tokenizer that only captures words
#Requires that input has been preprocessed to lowercase all letters
pattern = r'[\d.,]+|[A-Z][.A-Z]+\b\.*|\w+|\S'

TOKENIZER = RegexpTokenizer(r"\w+")


def countngrams(frequencies, order):

    stop_words = stopwords.words("dutch")
    newStopWords = ['nrc', 'nrc.nl','nu', 'nunl' 'nu.nl', 'telegraaf', 'telegraaf.nl', 'the', 'to', 'volkskrant', 'volkskrant.nl', 'dagelijksestandaard', '-', '„', '’', '','—', ':']

    stop_words.extend(newStopWords)

    articles = pd.read_csv('../csvbestanden/standaarddata' + '.csv', sep="|", error_bad_lines=False)
    articles['text_processed'] = \
    articles['Text'].map(lambda x: re.sub('[,\.!?]', '', x))

    articles['text_processed'] = \
    articles['text_processed'].map(lambda x: x.lower())
    articles['text_processed'] = articles['text_processed'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    text = articles['text_processed']
    # articles['Text'] = articles['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

    # text = articles['Text'].map(lambda x: x.lower())

    #Loop over the file while there is text to read
    nlp = spacy.load("nl_core_news_sm")
    excluded_tags = { "VERB", "ADP", "PUNCT", "NUM", "SYM", "AUX", "ADV", "CONJ", "DET", "PART", "PRON", "SCONJ", "X"}
    excluded_text = {"twitter", "frits", "bosch", "auteur", "volg", "michael", "dds", "eigenaar", "rtl", "lezers", "kanaal", "telegram", "gratis"}


    for i in text:
        k = ""
        # print(type(i))
        # maandlijst.clear()
        nlpdata = nlp(str(i))
        for s in nlpdata:
            if ((s.pos_ not in excluded_tags) and (s.is_stop is False) and (s.text not in excluded_text)):
                # print("erin")
                k+= str(s) + ', '


        spans = TOKENIZER.span_tokenize(k)
        # print(i)

        tokens = (k[begin : end] for (begin, end) in spans)
        for bigram in ngrams(tokens, order):
            print(bigram)
            #Increment the count for the bigram. Automatically handles any
            #bigram not seen before.
            frequencies[bigram] += 1


    return


if __name__ == '__main__':
    x=[]
    y=[]
    #Initialize the frequency distribution, a Subclass of collections.Counter
    frequencies = FreqDist()
    #The order of the ngrams is the first command line argument
    ngram_order = int(sys.argv[1])
    #Pull the input data from the console
    countngrams(frequencies, ngram_order)



    all_fdist = frequencies.most_common(50)
    bigram_df = pd.DataFrame(all_fdist, columns=['bigram', 'count'])

    dicts = bigram_df.set_index('bigram').T.to_dict('records')

    graph = nx.Graph()

    # Create connections between nodes
    for k, v in dicts[0].items():
        graph.add_edge(k[0], k[1], weight=(v * 10))

    d = nx.degree(graph)

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(graph, k=2)
    de = dict(graph.degree)
    de2 = [de[v]*100 for v in de.keys()]
    # nod = nx.get_node_attributes(graph, 'count')
    # Plot networks
    nx.draw_networkx(graph, pos, font_size=16, node_size = de2, width=3, edge_color='grey', node_color='purple', with_labels = False, ax=ax)
    # Create offset labels
    for key, value in pos.items():
        x,y = value[0], value[1]
        ax.text(x, y, s=key, horizontalalignment='center', fontsize=13)





















    #Choose colors for node and edge highlighting
node_highlight_color = 'white'
edge_highlight_color = 'black'

#Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
size_by_this_attribute = 'adjusted_node_size'
color_by_this_attribute = 'modularity_color'

#Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
color_palette = Blues8

#Choose a title!
title = 'Most common bigrams, De Dagelijkse Standaard'

#Establish which categories will appear when hovering over each node
HOVER_TOOLTIPS = [
       ("Character", "@index"),
        ("Degree", "@degree"),
         ("Modularity Class", "@modularity_class"),
        ("Modularity Color", "$color[swatch]:modularity_color"),
]

#Create a plot — set dimensions, toolbar, and title
plot = figure(tooltips = HOVER_TOOLTIPS,
              tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

#Create a network graph object
# https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
network_graph = from_networkx(G, networkx.spring_layout, scale=10, center=(0, 0))

#Set node sizes and colors according to node degree (color as category from attribute)
network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute, fill_color=color_by_this_attribute)
#Set node highlight colors
network_graph.node_renderer.hover_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)
network_graph.node_renderer.selection_glyph = Circle(size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2)

#Set edge opacity and width
network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.3, line_width=1)
#Set edge highlight colors
network_graph.edge_renderer.selection_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)
network_graph.edge_renderer.hover_glyph = MultiLine(line_color=edge_highlight_color, line_width=2)

    #Highlight nodes and edges
network_graph.selection_policy = NodesAndLinkedEdges()
network_graph.inspection_policy = NodesAndLinkedEdges()

plot.renderers.append(network_graph)

#Add Labels
x, y = zip(*network_graph.layout_provider.graph_layout.values())
node_labels = list(G.nodes())
source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
labels = LabelSet(x='x', y='y', text='name', source=source, background_fill_color='white', text_font_size='10px', background_fill_alpha=.7)
plot.renderers.append(labels)

show(plot)
#save(plot, filename=f"{title}.html")

    plt.show()
