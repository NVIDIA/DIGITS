# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

import json
import zipfile
from StringIO import StringIO

from sqlalchemy_backend import Blob, File, Model, Database, Job


class ModelStore:
    def __init__(self, db_file):
        self.db = Database(db_file)

    def models(self):
        session = self.db.get_session()
        model_list = list()
        for model in session.query(Model).all():
            tmp = model.get_dict()
            tmp_dict = dict()
            for tmp_file in model.files:
                tmp_dict[tmp_file.form_name] = (tmp_file.file_name, tmp_file.id)
            tmp['files'] = tmp_dict
            if model.job is not None:
                tmp['job'] = {
                    'image_dimension': [model.job.image_width, model.job.image_height, model.job.image_channel],
                    'image_resize_mode': model.job.image_resize_mode,
                    'framework': model.job.framework}
            model_list.append(tmp)
        self.db.return_session()
        return model_list

    def get_zip(self):
        session = self.db.get_session()
        s = StringIO()
        zf = zipfile.ZipFile(s, mode='w')
        model_names = dict()
        for model in session.query(Model).all():
            if model.name in model_names:
                count = model_names[model.name]
                model_names[model.name] = count + 1
                for tmp_file in model.files:
                    zf.writestr('{}_{}/{}'.format(model.name, count, tmp_file.file_name), tmp_file.blob.content)
            else:
                model_names[model.name] = 1
                for tmp_file in model.files:
                    zf.writestr('{}/{}'.format(model.name, tmp_file.file_name), tmp_file.blob.content)
        self.db.return_session()
        return s

    def get_models_zip(self, model_keys):
        session = self.db.get_session()
        s = StringIO()
        zf = zipfile.ZipFile(s, mode='w')
        model_names = dict()
        for model_key in model_keys:
            model = session.query(Model).filter_by(id=model_key).one()
            if model.name in model_names:
                count = model_names[model.name]
                model_names[model.name] = count + 1
                for tmp_file in model.files:
                    zf.writestr('{}_{}/{}'.format(model.name, count, tmp_file.file_name), tmp_file.blob.content)
            else:
                model_names[model.name] = 1
                for tmp_file in model.files:
                    zf.writestr('{}/{}'.format(model.name, tmp_file.file_name), tmp_file.blob.content)
        self.db.return_session()
        return s

    def get_one(self, model_key):
        session = self.db.get_session()
        s = StringIO()
        zf = zipfile.ZipFile(s, mode='w')
        model = session.query(Model).filter_by(id=model_key).one()
        model_name = model.name
        for tmp_file in model.files:
            zf.writestr(tmp_file.file_name, tmp_file.blob.content)
        self.db.return_session()
        return s, model_name

    def delete_one(self, model_key):
        session = self.db.get_session()
        session.delete(session.query(Model).filter_by(id=model_key).one())
        session.commit()
        self.db.return_session()
        return True

    def delete_models(self, model_keys):
        session = self.db.get_session()
        for model_key in model_keys:
            session.delete(session.query(Model).filter_by(id=model_key).one())
        session.commit()
        self.db.return_session()
        return True

    def commit(self):
        return True

    def get_model(self, model_key):
        session = self.db.get_session()
        model = session.query(Model).filter_by(id=model_key).one()
        result = {'name': model.name, 'notes': model.notes}
        tmp_list = list()
        for tmp_file in model.files:
            tmp_list.append({'form_name': tmp_file.form_name, 'file_name': tmp_file.file_name, 'file_key': tmp_file.id})
        result['files'] = tmp_list
        self.db.return_session()
        return result

    def get_file(self, file_key):
        session = self.db.get_session()
        file_info = session.query(File).filter_by(id=file_key).one()
        file_name = file_info.file_name
        file_content = file_info.blob.content
        self.db.return_session()
        return file_name, file_content

    def add(self, model):
        session = self.db.get_session()
        m = Model(model['name'], model['description'], model['instruction'], model['dataset'], model['license'])
        for f in model['files']:
            if f['file_name'] == 'info.json':
                tmp_dict = json.loads(f['content'])
                if 'image dimensions' in tmp_dict:
                    m.job = Job(tmp_dict['framework'], tmp_dict['image dimensions'],
                                tmp_dict['image resize mode'])
            tmp_file = File(f['form_name'], f['file_name'])
            tmp_file.blob = Blob(f['content'])
            m.files.append(tmp_file)
        session.add(m)
        model_key = m.id
        session.commit()
        self.db.return_session()
        return model_key
