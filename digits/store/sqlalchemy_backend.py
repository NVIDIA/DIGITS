# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

import uuid

from sqlalchemy import Column, create_engine, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session

# make the declarative base class for the ORM mapping
Base = declarative_base()

__model_version__ = 1


class Blob(Base):
    __tablename__ = 'blobs'

    id = Column(String, primary_key=True)
    content = Column(LargeBinary)
    file_id = Column(String, ForeignKey('files.id', ondelete='CASCADE'), index=True)
    file = relationship('File', back_populates='blob')

    def __init__(self, content):
        self.id = str(uuid.uuid4())
        self.content = content

    def __repr__(self):
        header = ''.join('{:02X}'.format(_) for _ in self.content[:8])
        tail = ''.join('{:02X}'.format(_) for _ in self.content[-8:])
        return 'Blob<id={}, content([:8], [-8:])=({},{})>'. \
            format(self.id, header, tail)


class File(Base):
    __tablename__ = 'files'

    id = Column(String, primary_key=True)
    file_name = Column(String)
    form_name = Column(String)
    model_id = Column(String, ForeignKey('models.id', ondelete='CASCADE'), index=True)
    model = relationship('Model', back_populates='files')
    blob = relationship('Blob', uselist=False, back_populates='file', cascade='all, delete, delete-orphan')

    def __init__(self, form_name=None, file_name=None):
        self.id = str(uuid.uuid4())
        self.form_name = form_name
        self.file_name = file_name

    def update(self, form_name=None, file_name=None, model_id=None):
        if form_name is not None and self.form_name != form_name:
            self.form_name = form_name
        if file_name is not None and self.file_name != file_name:
            self.file_name = file_name
        if model_id is not None and self.model_id != model_id:
            self.model_id = model_id

    def __repr__(self):
        return 'File<id={}, file_name={}, form_name={}>'. \
            format(self.id, self.file_name, self.form_name)


class Job(Base):
    __tablename__ = 'jobs'

    id = Column(String, primary_key=True)
    framework = Column(String)
    image_width = Column(Integer)
    image_height = Column(Integer)
    image_channel = Column(Integer)
    image_resize_mode = Column(String)
    model_id = Column(String, ForeignKey('models.id', ondelete='CASCADE'), index=True)
    model = relationship('Model', back_populates='job')

    def __init__(self, frmwk, img, im_rz_md):
        self.id = str(uuid.uuid4())
        self.framework = frmwk
        self.image_width = img[0]
        self.image_height = img[1]
        self.image_channel = img[2]
        self.image_resize_mode = im_rz_md

    def __repr__(self):
        return 'Job<id={}, framework={}, image dim={}, image resize mode={}>'. \
            format(self.id, self.framework,
                   'x'.join([self.image_width, self.image_height, self.image_channel]),
                   self.image_resize_mode)


class Model(Base):
    __tablename__ = 'models'

    id = Column(String, primary_key=True)
    name = Column(String)
    notes = Column(String)
    files = relationship('File', back_populates='model', cascade='all, delete, delete-orphan')
    job = relationship('Job', uselist=False, back_populates='model', cascade='all, delete, delete-orphan')

    def __init__(self, model_name, model_notes):
        self.id = str(uuid.uuid4())
        self.name = model_name
        self.notes = model_notes

    def update(self, name=None, notes=None):
        if name is not None and self.name != name:
            self.name = name
        if notes is not None and self.notes != notes:
            self.notes = notes

    def get_files(self):
        return [f for f in self.files]

    def remove_children(self):
        pass

    def __repr__(self):
        return 'Model<id={}, name={}, notes={}>'.format(self.id, self.name, self.notes)


class Database():
    def __init__(self, db_path):
        self.engine = create_engine('sqlite:///' + db_path)
        Base.metadata.create_all(self.engine)
        self.SCSession = scoped_session(sessionmaker(bind=self.engine))

    def close(self):
        self.engine.dispose()
        return True

    def get_session(self):
        return self.SCSession()

    def return_session(self):
        self.SCSession.remove()
        return True
