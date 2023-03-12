from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import Email, EqualTo, ValidationError, AnyOf, InputRequired, Length, Optional

from app.models import User

CURRENCIES = ['USD', 'EUR', 'GBP', 'AUD', 'NZD', 'CAD', 'JPY']


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    email = StringField('Email', validators=[InputRequired(), Email()])
    password = PasswordField('Password', validators=[
                             InputRequired(), Length(min=8, max=200)])
    password2 = PasswordField('Repeat Password', validators=[
                              InputRequired(), EqualTo('password')])
    currency = StringField('Currency (3-letter)',
                           validators=[AnyOf(CURRENCIES)])
    submit = SubmitField('Sign In')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError(
                'Username taken. Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError(
                'Email address taken. Please use a different email address.')


class UpdateDetailsForm(FlaskForm):
    email = StringField('Email', validators=[Email()])
    existing_password = PasswordField(
        'Existing Password', validators=[InputRequired(message="Please insert existing password to update details")])
    password = PasswordField('New Password', validators=[
                             Length(min=8, max=200), Optional()])
    password2 = PasswordField(
        'Repeat Password', validators=[EqualTo('password')])
    currency = StringField('Currency (3-letter)',
                           validators=[AnyOf(CURRENCIES)])
    submit = SubmitField('Update Details')
