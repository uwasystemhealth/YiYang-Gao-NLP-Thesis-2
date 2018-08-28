import csv
import re
import os
from collections import defaultdict

trial_number = 1
save_folder_name = "./Input_Output_Folder/Preprocessed_Record/"
if not os.path.isdir(save_folder_name):
    os.makedirs(save_folder_name)

ignored_lines_placeholder = '='

class Words_Parser(object):
    def __init__(self, dir_name , file_name ):
        self.file_name = file_name
        self.dir_name  = dir_name

    def __iter__(self):

        with open(self.dir_name + self.file_name , newline='') as words_file:
            for line  in words_file:
                yield line.split()[1]

class Sentences_Parser_3(object):
    def __init__(self, file_name):
        self.file_name = file_name
        with open(self.dir_name + self.file_name , newline='') as words_file:
            for line  in words_file:
                yield line.split()[1]

class Sentences_Parser_3(object):
    def __init__(self, file_name):
        self.file_name = file_name


    def __iter__(self):

        with open(self.file_name, "r", newline='') as data_file:
                line = data_file.readline()
                while line:
                    s = line.split('\t')
                    if len(s) > 1:
                        yield s[1]
                    line = data_file.readline()



class Sentences_Parser_2(object):
    def __init__(self, file_name):
        self.file_name = file_name

    def __iter__(self):

        with open(self.file_name, "r", newline='') as data_file:
                line = data_file.readline()
                while line:
                    if line.__contains__('~'):
                        line = data_file.readline()
                    else:
                        s = line.split('\t')
                        if len(s) > 1 :
                            yield s[1].split()
                        line = data_file.readline()



class Sentences_Parser(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.ignored_lines_count = 0
        self.total_lines = 0
        self.total_number_of_words = 0

        self.Match_Case_1 = re.compile(r'((\w*[\w\s])\/([\w\s]\w*))')                  #for words connected wby /
        self.Match_Case_2 = re.compile(r'([\s|^][a-z]{2,4}[\d|\s]\d{1,4})[\s|.]')      #for equipment ids
        self.Match_Case_3 = re.compile(r'\s[0-9]{2}[/.][0-9]{2}[/.][0-9]{2,4}')              #for datatiem format
        self.Match_Case_4 = re.compile('(?<=\d)[\.\/](?=\d)')                          # for cases like 12.25 or 12/25
        self.Match_Case_5 = re.compile('(?<![a-z]{2,2})\.(?![a-z]{2,20})')

    def Process_short_text(self, short_text):
        short_text = '  ' + short_text + '  '                                          # append whitespace at the back and front for easy of regex matching

        short_text = re.sub('&', ' and ', short_text)
        short_text = re.sub('@', ' at ', short_text)
        short_text = re.sub('\[', ' ', short_text)                                   #[safas] get rid of brackets
        short_text = re.sub('\]', ' ', short_text)                                   #[safas] get rid of brackets
        short_text = re.sub('re-', 're', short_text)                                # change re-kit to 'rekit'
        short_text = re.sub('-', ' ', short_text)
        short_text = re.sub('no.[\d|\s]\d+', ' number ', short_text)
        short_text = re.sub('\'\s?s\s', ' ', short_text)                                  #get rid of cases like drive's --> driver
        short_text = re.sub('\'\s?t\s', 't ', short_text)                               # get rid of cases like isn't --> isnt
# ------------------------third match Case---------------------------------------------------------------------------------------------------------------
        short_text = re.sub(self.Match_Case_3.pattern, ' date_time ', short_text)

# ------------------------fourth match Case---------------------------------------------------------------------------------------------------------------
        short_text = re.sub(self.Match_Case_4.pattern, '', short_text)
        # delete the . or / inbetween two numebrs
        # e.g.  12.23 ---> 1223
        # e.g.  12/23 ---> 1223

#-------------------------First Case---------------------------------------------------------------------------------------------------------------
        # found_string_list = self.Match_Case_1.findall(short_text)
        # for found_string in found_string_list:
        #     if len(found_string) > 7:
        #         short_text = re.sub('\w\/\w', found_string[0].replace('/', ' '), short_text)
        #     if len(found_string) > 0 and len(found_string) < 7:
        #         short_text = re.sub('\w\/\w', found_string[0].replace('/', '_'), short_text)
        found_string_list = self.Match_Case_1.findall(short_text)
        for found_string in found_string_list:
            t0 = found_string[1].strip()
            t1 = found_string[2].strip()
            if len(t0) + len(t1) > 7:
                string_to_replace = t0 + ' ' + t1
            else:
                string_to_replace = t0 + '_' + t1
            #print(string_to_replace)
            short_text = re.sub(self.Match_Case_1.pattern, string_to_replace, short_text)

#-------------------------Second Case---------------------------------------------------------------------------------------------------------------
        short_text = re.sub(self.Match_Case_2.pattern ,' equipment_id ' , short_text )


# ------------------------fifth match Case---------------------------------------------------------------------------------------------------------------
        short_text = re.sub(self.Match_Case_5.pattern, '', short_text)
        # delete the dot inbetween individual chars for abbreviation
        # e.g.  e.m.s ---> ems
        # e.g.  p.s.i ---> psi

#-------------------------general case----------------------------------------------------------------------------------------------
        short_text = re.sub(r"'+", '', short_text)

        short_text = re.sub('(?<=\d)\s?x\s?(?=\d)', ' x ', short_text)   # seperate cases like 2x3
        short_text = re.sub('\d+', '_number_', short_text)

        short_text = re.sub('[^_\w]+', ' ', short_text)    #replace charter other than '_'


#--------------------------Speacial Cases---------------------------------------------------------------------------------------------------
        short_text = re.sub('pos_number_', ' pos _number_  ', short_text)

        short_text = re.sub('(edd[_|_\s|\s_]number_)', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('edd _number_', ' equipment_id ', short_text)  # replace 'edd _number_' as 'equipment_id'
        short_text = re.sub('(tkd[_|_\s|\s_]number_)', ' equipment_id ',short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('tkd _number_', ' equipment_id ', short_text)  # replace 'edd _number_' as 'equipment_id'
        short_text = re.sub('exld_number_', ' equipment_id ', short_text)  # replace 'edd _number_' as 'equipment_id'
        short_text = re.sub(r'(she(\s_|_)number_)', ' equipment_id ',short_text)  # replace 'she_number_' as 'equipment_id'
        short_text = re.sub(r'(exd(\s_|_|l_)number_)', ' equipment_id ',short_text)  # replace 'she_number_' as 'equipment_id'
        short_text = re.sub(r'(tkw(\s_|_)number_)', ' equipment_id ',short_text)  # replace 'she_number_' as 'equipment_id'
        short_text = re.sub(r'(shd(\s_|_)number_)', ' equipment_id ',short_text)  # replace 'she_number_' as 'equipment_id'
        short_text = re.sub(r'(trd(\s_|_)number_)', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub(r'(ldw(\s_|_)number_)', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'

        short_text = re.sub('trw_number_', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('dmm_number_', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('tks_number_', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('dre_number_', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('cid_number_', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('cat_number_\w?', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('evv_number_', ' equipment_id ', short_text)  # replace 'edd_number_' as 'equipment_id'
        short_text = re.sub('grd\/?_number_', ' equipment_id ', short_text)  # replace 'edd _number_' as 'equipment_id'
        short_text = re.sub(r'(rd(\s_|_)number_)', ' equipment_id ',short_text)  # replace 'edd_number_' as 'equipment_id'

        short_text = re.sub('\sr and r\s', ' r_r ', short_text)                   # replace 'r and r' as 'r_r'
        short_text = re.sub('\sr and i\s', ' r_i ', short_text)                   # replace 'r and i' as 'r_r'
        short_text = re.sub('\sp and t\s', ' p_t ', short_text)  # replace 'r and i' as 'r_r'
        short_text = re.sub('\sp and h\s', ' p_h ', short_text)  # replace 'r and i' as 'r_r'
        short_text = re.sub('\st and c\s', ' t_c ', short_text)  # replace 't and c' as 't_c'

        short_text = re.sub('_number_\s?wkly', ' _number_ weekly ', short_text)
        short_text = re.sub('_number_\s?wk', ' _number_ week ', short_text)

        short_text = re.sub('_number_\s?yr', ' _number_ yearly ', short_text)
        short_text = re.sub('wk_?_number_', 'week _number_ ', short_text)

        short_text = re.sub('_number_\s?year', ' _number_ yearly ', short_text)
        short_text = re.sub('_number_\s?mth\s', ' _number_ month ', short_text)
        short_text = re.sub('_number_\s?mthly\s', ' _number_ monthly ', short_text)
        short_text = re.sub('_number_m', ' _number_ month ', short_text)
        short_text = re.sub('_number_\s?hr[\s|s]', ' _number_ hour ', short_text)
        short_text = re.sub('_number_\s?hour', ' _number_ hour ', short_text)
        short_text = re.sub('_number_h ', ' _number_ hour ', short_text)
        short_text = re.sub('_number_\s?min[\s|s] ', ' _number_ hour ', short_text)
        short_text = re.sub('_number_\s?sec ', ' _number_ sec ', short_text)

        short_text = re.sub(' _number_x(?![_x])', ' _number_ x ', short_text)

        short_text = re.sub('_number_kg ', ' _number_ kg ', short_text)
        short_text = re.sub('_number_volt ', ' _number_ volts ', short_text)
        short_text = re.sub('_number_v ', ' _number_ volts ', short_text)

        short_text = re.sub('_number_kv ', ' _number_ kilovolts ', short_text)
        short_text = re.sub('_number_w', ' _number_ watts ', short_text)
        short_text = re.sub('_number_kw', ' _number_ kilowatts ', short_text)
        short_text = re.sub('_number_psi', ' _number_ psi ', short_text)
        short_text = re.sub('_number_amp', ' _number_ amp ', short_text)
        short_text = re.sub('_number_deg', ' _number_ deg ', short_text)
        short_text = re.sub('_number_rpm', ' _number_ rpm ', short_text)
        short_text = re.sub('_number_t', ' _number_ ton ', short_text)
        short_text = re.sub('_number_litres', ' _number_ litres ', short_text)

        short_text = re.sub('_number__way', ' two_way ', short_text)

        short_text = re.sub('model_number_', 'model _number_ hour ', short_text)
        short_text = re.sub('manufacturer_number_', 'manufacturer _number_ hour ', short_text)

        short_text = re.sub('\s_number_st\s', ' first ', short_text)
        short_text = re.sub('\s_number_nd\s', ' second ', short_text)
        short_text = re.sub('\sreplace_number_\s', ' replace _number_ ', short_text)
        short_text = re.sub('\sweek_number_\s', ' week _number_ ', short_text)


        short_text = re.sub('\s\w?_number_\w?_number_\w?\s', ' ', short_text)
        short_text = re.sub('\s\w{1,2}_number_\w?\s', ' ', short_text)

        #------------------------------Speacial Case 2----------------------------------------------------------------------------------------------------------------

        short_text = re.sub('\som\s', ' on ', short_text)
        short_text = re.sub('\swill not\s',   ' cannot ', short_text)
        short_text = re.sub('wont\s',         ' cannot ', short_text)
        short_text = re.sub('won t\s',        ' cannot ', short_text)

        short_text = re.sub('cant\s',         ' cannot ', short_text)
        short_text = re.sub('can t\s',        ' cannot ', short_text)
        short_text = re.sub('\scan not\s',    ' cannot ', short_text)

        short_text = re.sub('\scould not\s',  ' cannot ', short_text)
        short_text = re.sub('\scouldnt\s',    ' cannot ', short_text)
        short_text = re.sub('\scouldn t\s',   ' cannot ', short_text)


        short_text = re.sub('\swould not\s', ' cannot ', short_text)
        short_text = re.sub('\swouldnt\s',     ' cannot ', short_text)
        short_text = re.sub('wouldn t\s',     ' cannot ', short_text)

        short_text = re.sub('\sdo not\s', ' cannot ', short_text)
        short_text = re.sub('\sd not\s', ' cannot ', short_text)
        shokcxrt_text = re.sub('\sdont\s', ' cannot ', short_text)

        short_text = re.sub('\sdoes not\s', ' cannot ', short_text)
        short_text = re.sub('\sdoesnt\s', ' cannot ', short_text)

        short_text = re.sub('\snot able to\s', ' cannot ', short_text)
        short_text = re.sub('\sunable to\s', ' cannot ', short_text)


# deal with common bigrams
        short_text = re.sub('over haul\s', ' overhaul ', short_text)
        short_text = re.sub('change out\s', ' changeout ', short_text)
        short_text = re.sub('up grade\s', ' upgrade ', short_text)
        short_text = re.sub('u s\s', ' u_s ', short_text)
        short_text = re.sub('\sa c\s', ' a_c ', short_text)
        short_text = re.sub('\sc b\s', ' c_b ', short_text)
        short_text = re.sub('\sh i d\s', ' h_i_d ', short_text)
        short_text = re.sub('\sl e d\s?s?\s', ' led ', short_text)
        short_text = re.sub('\sh m u\s', ' hmu ', short_text)
        short_text = re.sub('\sg u i\s', ' gui ', short_text)
        short_text = re.sub('\sp t o\s', ' p_t_o ', short_text)
        short_text = re.sub('\sg p s\s', ' g_p_s ', short_text)
        short_text = re.sub('\sp l c\s', ' plc ', short_text)
        short_text = re.sub('\sn c s\s', ' ncs ', short_text)
        short_text = re.sub('\sweek ys\s', ' weekly ', short_text)
        # ------------------------------Speacial Case 3----------------------------------------------------------------------------------------------------------------
        # deal with single char in the text
        short_text = re.sub('\sj (?=box)', ' j_', short_text)
        short_text = re.sub('\sj (?=bloc)', ' j_', short_text)
        short_text = re.sub('\sj (?=mod)', ' j_', short_text)

        short_text = re.sub('\se (?=stop)', ' emergency_', short_text)
        short_text = re.sub('\so (?=ring)', ' o_', short_text)
        short_text = re.sub('\sc (?=breaker)', ' circuit ', short_text)
        short_text = re.sub('\sr h\s', ' r_h ', short_text)
        short_text = re.sub('\sl h\s', ' r_h ', short_text)
        short_text = re.sub('(?<=_h)\ss\s', '_s ', short_text)
        short_text = re.sub('(?<=_h)\sf\s', '_f ', short_text)
        short_text = re.sub('(?<=_h)\sr\s', '_r ', short_text)
        short_text = re.sub('\sl and r\s', 'left and right ', short_text)
# ---------------------------------------------------------------------------------------------------------------------------------
        return short_text

    def __iter__(self):

        with open(self.dir_name + 'loader_work_orders_sanitised.csv', newline='') as csvfile:
            data_reader = csv.reader(csvfile)
            for row in data_reader:
                self.total_lines+=1
                short_text = row[4]
                # only process if contains space, to get rid of the records where there is no space seperation
                if short_text.find(' ') > 0 :
                    short_text = self.Process_short_text(short_text)
                    yield short_text.split()
                else:
                    self.ignored_lines_count+=1
                    yield ignored_lines_placeholder


def Precrocess_Records_and_Print(sentences):
    Path_to_Save_file = save_folder_name + 'Cleaned_Data_' + str(trial_number) +'.txt'

    with open(Path_to_Save_file, "w") as cleaned_data_file:
        i = 1
        for sentence in sentences:
            string_to_print = str(i) + '\t' + " ".join(sentence)
            print( string_to_print , file=cleaned_data_file)
            i = i +1

        print('*******************************************************************************', file=cleaned_data_file)
        string_to_print = 'Total:' + str(sentences.total_lines) + '\t' + 'Ignored:' + str(sentences.ignored_lines_count)
        print(string_to_print, file=cleaned_data_file)


def Print_Out_Token_Frequecy(sentences):
    frequency = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            frequency[word] += 1
            #sentences.total_number_of_words+=1
    i = 1
    with open("./Words_Correction_Dictionary/Cleaned_Data_Frequency_16.txt", "w") as cleaned_data_file:
        for w in sorted(frequency, key=frequency.get, reverse=True):
            string_to_print = str(i) + '\t' + w +'\t\t\t\t\t'+ str(frequency[w])
            print( string_to_print , file=cleaned_data_file)
            i+=1
        print('*******************************************************************************', file=cleaned_data_file)
        #string_to_print = 'Total:' + str(sentences.total_number_of_words)
        print(string_to_print, file=cleaned_data_file)
    return frequency

def Build_Frequency_Dic(sentences):
    frequency = defaultdict(int)
    for sentence in sentences:
        for word in sentence:
            frequency[word] += 1
    return frequency


def write_vacab_to_txt(file_path, words):
    with open(file_path, "w") as vacab_file:
        i = 1
        for word in words:
            string_to_print = '{0:15} \t {1:15}'.format(str(i),word)
            print(string_to_print, file = vacab_file)
            i = i + 1


if __name__ == "__main__":
    Path_to_Rawdata = './Input_Output_Folder/Raw_Record/'
    sentences = Sentences_Parser(Path_to_Rawdata)
    Precrocess_Records_and_Print(sentences)


    #sentences = Sentences_Parser_2('./Words_Correction_Dictionary/Normalized_Text_Stage_2.txt')
    #for s in sentences:
       #print(s)
    #Print_Out_Token_Frequecy(sentences)

